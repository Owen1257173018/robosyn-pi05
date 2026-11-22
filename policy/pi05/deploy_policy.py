# ----------------------------------------------------------------------------
# π₀ Policy Adapter for RoboSynChallenge
#
# 遵循 RoboTwin 统一评估接口:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os
import sys
import time

import numpy as np
import torch

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.insert(0, parent_directory)

from episode_timing import EpisodeTimer
from pi_model import GRIPPER_DIMS, PI0
from policy.inference_timing import finish_inference, start_inference


class EnvActionFormat:
    """Converts a pi0 action chunk into the torch actions EmbodiChain accepts, one (1, action_dim) tensor per step.

    The env's action space and device are read once per episode (eval() creates this on the first call after
    reset_model), and the whole chunk is converted at once, so executing an action costs one env.step() and one
    success check and nothing else.

    The policy outputs the gripper as a normalized opening (0 closed, 1 open), the unit of the Real data. With
    `rescale_gripper` it is mapped into the env's gripper action range, as the official SmolVLA adapter does: the
    simulator takes the finger position in meters (0-0.05), so without it every output above 0.05 is fully open.
    A gripper whose declared range is not finite is sent unchanged.
    """

    def __init__(self, env, rescale_gripper=False):
        unwrapped = env.unwrapped
        action_space = unwrapped.single_action_space
        self.dim = int(np.prod(action_space.shape))
        self.device = unwrapped.device
        self.gripper_ranges = []
        if rescale_gripper:
            low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)[GRIPPER_DIMS]
            high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)[GRIPPER_DIMS]
            self.gripper_ranges = [
                (dim, lo, hi)
                for dim, lo, hi in zip(GRIPPER_DIMS, low, high)
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo
            ]

    def __call__(self, actions):
        chunk = np.asarray(actions, dtype=np.float32)
        chunk = chunk.reshape(chunk.shape[0], -1)
        if chunk.shape[1] < self.dim:
            raise ValueError(f"Policy action has dim {chunk.shape[1]}, but env expects {self.dim}.")
        chunk = chunk[:, : self.dim].copy()
        for dim, lo, hi in self.gripper_ranges:
            chunk[:, dim] = lo + np.clip(chunk[:, dim], 0.0, 1.0) * (hi - lo)
        if not np.isfinite(chunk).all():
            bad = sorted({int(i) for i in np.argwhere(~np.isfinite(chunk))[:, 1]})
            raise RuntimeError(f"pi05 produced non-finite actions in dims {bad}; nothing of this chunk was sent.")
        tensor = torch.as_tensor(chunk, dtype=torch.float32, device=self.device)
        return [tensor[i : i + 1] for i in range(tensor.shape[0])]


def encode_obs(obs):
    """Convert gym Gymnasium Dict observation to π₀ input format.

    EmbodiChain observation keys:
        "sensor/cam_high/color"        -> base camera
        "sensor/cam_left_wrist/color"  -> left wrist
        "sensor/cam_right_wrist/color" -> right wrist
        "robot/qpos"                   -> joint state

    Returns:
        img_arr:  list of [img_front, img_right, img_left] as (H, W, C) numpy arrays
        state:    joint state vector
    """
    try:
        img_front_raw = obs["sensor"]["cam_high"]["color"]
        img_left_raw = obs["sensor"]["cam_left_wrist"]["color"]
        img_right_raw = obs["sensor"]["cam_right_wrist"]["color"]
        state_raw = obs["robot"]["qpos"]
    except (KeyError, TypeError) as error:
        raise KeyError(
            "pi05 needs obs['sensor'][cam]['color'] for cam_high, cam_left_wrist, cam_right_wrist and "
            f"obs['robot']['qpos']; missing {error}"
        ) from error

    img_front = img_front_raw[0, ..., :3]
    img_left = img_left_raw[0, ..., :3]
    img_right = img_right_raw[0, ..., :3]

    # Joint state — (num_envs, num_joints) -> squeeze env dim
    state = state_raw[0]
    img_arr = [img_front, img_right, img_left]

    return img_arr, state


def _flag(usr_args, key, default=False):
    """A boolean argument; the strings "false", "0", "no" and "none" count as false (bool("false") is True)."""
    value = usr_args.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "none", "null")
    return bool(value)


def get_model(usr_args):
    """Create the π0.5 policy of one task.

    Fields of usr_args (deploy_policy.yml plus the overrides):
        task_name          — one of the 10 tasks; selects the task's openpi config (TRAIN_CONFIGS in pi_model.py),
                             its training prompt and the start-of-episode gripper rule
        checkpoint_path    — the task's checkpoint folder, i.e. the Hugging Face folder <task_name>/ (params/, assets/);
                             default policy/pi05/checkpoints/<task_name>. `checkpoint_dir` is accepted as well.
        train_config_name  — optional: another openpi config than the task's
        pi0_step           — actions executed per inference (the model predicts 50)
        warm_up            — compile the model at load time (default true)
        episode_time_limit_s — wall-clock budget of an episode; eval() then returns truncated=True (null: none)
    """
    model = PI0(
        task_name=usr_args.get("task_name"),
        checkpoint_dir=usr_args.get("checkpoint_path") or usr_args.get("checkpoint_dir"),
        train_config_name=usr_args.get("train_config_name"),
        pi0_step=int(usr_args.get("pi0_step", 50)),
        pytorch_device=usr_args.get("pytorch_device", "cuda"),
        open_grippers_at_start=_flag(usr_args, "open_grippers_at_start"),
        close_drawer_after_place=_flag(usr_args, "close_drawer_after_place"),
        action_samples=int(usr_args.get("action_samples", 1)),
        noise_scale=float(usr_args.get("noise_scale", 1.0)),
    )
    model.rescale_gripper = _flag(usr_args, "rescale_gripper")
    model.timer = EpisodeTimer(model.task_name)
    model.env_action_format = None
    limit = usr_args.get("episode_time_limit_s")
    model.episode_time_limit_s = float(limit) if _flag(usr_args, "episode_time_limit_s") else None
    if _flag(usr_args, "warm_up", True):
        model.warm_up()
    return model


def eval(env, model, obs):
    """Run one inference and execute its action chunk in the environment.

    1. On the first call of an episode: set the instruction and read the env's action space once.
    2. Read the observation into the model input and infer one chunk (`pi0_step` actions).
    3. Execute the chunk in order: per action one env.step() and one success check (dropped for the rest of the
       episode if is_task_success() turns out slow; the harness checks success after every call anyway). The
       observation is read again only by the next call (from the obs the harness passes back), except with the
       simulator-only drawer rule, which needs the joint state of every step.
    """
    timer = model.timer
    timer.ensure_started()
    eval_started = time.perf_counter()
    if model.observation_window is None:
        instruction = getattr(env, "_current_instruction", None)
        model.set_language(instruction)
        model.env_action_format = EnvActionFormat(env, model.rescale_gripper)

    inference_times_s = []
    started_at = start_inference(model.pytorch_device)
    t0 = time.perf_counter()
    img_arr, state = encode_obs(obs)
    model.update_observation_window(img_arr, state)
    t1 = time.perf_counter()
    actions = model.get_action()[: model.pi0_step]
    t2 = time.perf_counter()
    action_tensors = model.env_action_format(actions)
    finish_inference(started_at, inference_times_s, model.pytorch_device)
    timer.chunk(encode_s=t1 - t0, model_s=t2 - t1, format_s=time.perf_counter() - t2)

    # The drawer rule (simulator only) replays the right arm's joint history, so it needs every step's state.
    per_step_obs = model.drawer_closer is not None
    final_obs = obs
    info = None
    truncated = False
    end_reason = None
    for action_tensor in action_tensors:
        step_started = timer.before_step()
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        # The `gym_config` setting configures the `actionmanager` to support delta action input;
        # the default action must be absolute `qpos`.
        step_done = time.perf_counter()
        # Checked after every step while that is cheap (it is free in the simulator); see episode_timing.py.
        success, success_s = False, None
        if timer.check_each_step:
            success = env.get_wrapper_attr("is_task_success")()
            success_s = time.perf_counter() - step_done
        timer.after_step(step_done - step_started, success_s)

        if success:
            end_reason = "success"
            break
        # The time limit wrapper returns a plain bool once the step limit is reached, the env a tensor otherwise.
        if torch.as_tensor(truncated).any():
            end_reason = "step limit"
            break
        # Optional wall-clock budget (`episode_time_limit_s`): end the episode as a timeout, once, instead of
        # running into the evaluation's own time limit.
        if timer.over_limit(model.episode_time_limit_s):
            truncated = True
            end_reason = f"time limit {model.episode_time_limit_s:.0f} s"
            break

        if per_step_obs:
            obs_started = time.perf_counter()
            img_arr, state = encode_obs(final_obs)
            model.update_observation_window(img_arr, state)
            timer.obs_per_step += time.perf_counter() - obs_started

    timer.eval_done(time.perf_counter() - eval_started, end_reason)
    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    """Reset π₀ internal state (observation window and instruction) and start the episode's timing."""
    model.reset_obsrvationwindows()
    model.env_action_format = None
    model.timer.start()
