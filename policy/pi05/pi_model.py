"""π0.5 policy wrapper used by deploy_policy.py (adapted from policy/pi0/pi_model.py)."""

import dataclasses
import os

import numpy as np
from drawer_closer import DrawerCloser
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

PI05_ROOT = os.path.dirname(os.path.abspath(__file__))

# Gripper dims of the 14-dim action. The policy is trained on the normalized opening (0 closed, 1 open);
# outputs below the threshold are sent as 0, because the sim env reads the gripper in meters (open >= 0.05 m)
# and a residual such as 0.02 would otherwise half-open a grasp.
GRIPPER_DIMS = [6, 13]
GRIPPER_CLOSE_BELOW = 0.25

# (left, right) grippers the demonstrations open at the start of an episode. The sim env resets with both grippers
# closed, while the Sim data opens them within one frame (or starts open: drawer, assembly) and the Real data
# mostly starts open, so the policy keeps copying the closed state; with `open_grippers_at_start` they are opened
# for the first action chunk. click_bell presses with closed fingers, mixer / pipette keep the left one closed.
# table_rearrangement opens after ~13 frames, which the policy already learned.
OPEN_AT_START = {
    "handle_basket": (True, True),
    "water_pouring": (True, True),
    "items_handover": (True, True),
    "drawer_open_place": (True, True),
    "mixer_operating": (False, True),
    "item_assembly": (True, True),
    "manipulate_pipette": (False, True),
    "sample_loading": (True, True),
}

# Prompts the policy is trained with (`meta/tasks.jsonl`, identical in the official Sim and Real datasets).
TASK_INSTRUCTIONS = {
    "click_bell": "Click the bell",
    "handle_basket": "Pick the bottle then place it to the basket, carry the basket with the other arm.",
    "water_pouring": "Pick the cup and the bottle with the other arm, pour the water from bottle to cup",
    "table_rearrangement": "Pick the fork and the spoon, place them next to the plate.",
    "items_handover": "Pick up the pen, hand it over to the other arm and then place it in to the pen holder",
    "drawer_open_place": "Open the drawer, pick the tomato with the other arm then place it in the drawer.",
    "mixer_operating": "Pick the beaker, place it on the mixer, then flip the toggle switch with the other arm.",
    "item_assembly": "Pick up two centrifuge tubes from the table and dock them horizontally.",
    "manipulate_pipette": "Use the pipette on the table to transfer liquid from one beaker to another",
    "sample_loading": "Pick up the test tube, and it to the other arm, and insert it to the rack.",
}

# openpi training config of each task's checkpoint (src/openpi/training/config.py, README §1). It defines the model,
# its input/output transforms and the folder of the norm stats under the checkpoint's assets/.
# table_rearrangement and item_assembly use the same checkpoint, trained on all 10 tasks.
TRAIN_CONFIGS = {
    "click_bell": "pi05_robosyn_v3nf_click_bell_gen",
    "handle_basket": "pi05_robosyn_v3nf_handle_basket_gen",
    "water_pouring": "pi05_robosyn_v3nf_water_pouring_gen2only",
    "table_rearrangement": "pi05_robosyn_v3nf_r50",
    "items_handover": "pi05_robosyn_v3nf_items_handover_gen",
    "drawer_open_place": "pi05_robosyn_v3nf_drawer_open_place_gen",
    "mixer_operating": "pi05_robosyn_v3nf_mixer_operating_gen",
    "item_assembly": "pi05_robosyn_v3nf_r50",
    "manipulate_pipette": "pi05_robosyn_v3nf_manipulate_pipette_gen2",
    "sample_loading": "pi05_robosyn_v3nf_r50up",
}


def resolve_checkpoint_dir(checkpoint_path, task_name):
    """Absolute path of the task's checkpoint folder (the Hugging Face folder <task_name>/: params/, assets/).

    Without a path it is policy/pi05/checkpoints/<task_name>. A relative path that does not exist from the current
    directory is also looked up from policy/pi05, so `checkpoints/<task_name>` works from the repository root too.
    """
    if not checkpoint_path:
        if not task_name:
            raise ValueError("set task_name or checkpoint_path")
        checkpoint_path = os.path.join(PI05_ROOT, "checkpoints", task_name)
    path = os.path.expanduser(str(checkpoint_path))
    if not os.path.isabs(path) and not os.path.exists(path) and os.path.exists(os.path.join(PI05_ROOT, path)):
        path = os.path.join(PI05_ROOT, path)
    path = os.path.abspath(path)
    if not os.path.isdir(os.path.join(path, "params")):
        raise FileNotFoundError(
            f"no checkpoint at {path} (it needs params/ and assets/): download the Hugging Face folder "
            f"{task_name or '<task_name>'}/ of owen1257173018/robosyn-pi05 (policy/pi05/README.md §2)"
        )
    return path


class PI0:

    def __init__(
        self,
        task_name,
        checkpoint_dir=None,
        train_config_name=None,
        pi0_step=50,
        pytorch_device="cuda",
        open_grippers_at_start=False,
        close_drawer_after_place=False,
        action_samples=1,
        noise_scale=1.0,
    ):
        self.task_name = task_name
        self.train_config_name = train_config_name or TRAIN_CONFIGS.get(task_name)
        if self.train_config_name is None:
            raise ValueError(
                f"unknown task_name {task_name!r}: use one of {', '.join(sorted(TRAIN_CONFIGS))}, or set train_config_name"
            )
        self.pytorch_device = pytorch_device
        self.checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir, task_name)
        print(f"task {task_name}: openpi config {self.train_config_name}, checkpoint {self.checkpoint_dir}")
        self.open_at_start = np.array(OPEN_AT_START.get(task_name, (False, False)), dtype=bool) & open_grippers_at_start
        self.episode_start = True
        # Proprioception of every control step of the episode, for the drawer rule.
        self.state_history = []
        self.drawer_closer = DrawerCloser() if close_drawer_after_place and task_name == "drawer_open_place" else None
        # Action chunks sampled per inference and averaged; each infer() call draws new flow-matching noise.
        self.action_samples = max(1, int(action_samples))
        # Scale of the flow-matching noise each chunk is sampled from: at 1 the policy draws the trained N(0, I)
        # itself; smaller values sample closer to the most likely chunk, 0 deterministically from zero noise.
        self.noise_scale = float(noise_scale)
        model_config = self._config_for_device(pytorch_device).model
        self.noise_shape = (model_config.action_horizon, model_config.action_dim)
        self.noise_rng = np.random.default_rng(0)

        self.policy = self._create_policy(self.pytorch_device)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        self.pi0_step = pi0_step

    def _config_for_device(self, pytorch_device):
        config = _config.get_config(self.train_config_name)
        if pytorch_device == "cpu" and hasattr(config.model, "pytorch_compile_mode"):
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, pytorch_compile_mode=None),
            )
        return config

    def _create_policy(self, pytorch_device):
        return _policy_config.create_trained_policy(
            self._config_for_device(pytorch_device),
            self.checkpoint_dir,
            pytorch_device=pytorch_device,
        )

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # Use the training prompt of the task; the environment's instruction is only a fallback.
    def set_language(self, instruction):
        canonical = TASK_INSTRUCTIONS.get(self.task_name)
        if canonical is not None and instruction != canonical:
            print(f"instruction {instruction!r} replaced by training prompt {canonical!r}")
            instruction = canonical
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]

        self.observation_window = {
            "observation/image": img_front,
            "observation/left_wrist_image": img_left,
            "observation/right_wrist_image": img_right,
            "observation/state": state,
            "prompt": self.instruction,
        }
        self.state_history.append(np.array(state, dtype=np.float64).reshape(-1))

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        # deploy_policy.eval passes the last observation of the previous chunk again before inferring.
        if len(self.state_history) > 1 and np.array_equal(self.state_history[-1], self.state_history[-2]):
            self.state_history.pop()
        samples = [
            np.array(self.policy.infer(self.observation_window, **self._noise())["actions"])
            for _ in range(self.action_samples)
        ]
        actions = samples[0] if len(samples) == 1 else np.mean(samples, axis=0)
        gripper = actions[:, GRIPPER_DIMS]
        actions[:, GRIPPER_DIMS] = np.where(gripper < GRIPPER_CLOSE_BELOW, 0.0, gripper)
        if self.episode_start and self.open_at_start.any():
            closed = np.asarray(self.observation_window["observation/state"])[GRIPPER_DIMS] < GRIPPER_CLOSE_BELOW
            for dim in np.asarray(GRIPPER_DIMS)[closed & self.open_at_start]:
                actions[:, dim] = 1.0
        self.episode_start = False
        if self.drawer_closer is not None and self.state_history:
            actions = self.drawer_closer(actions, np.stack(self.state_history), self.pi0_step)
        return actions

    # infer() kwargs for one chunk: none at noise scale 1, else the scaled noise.
    def _noise(self):
        if self.noise_scale == 1.0:
            return {}
        return {"noise": (self.noise_scale * self.noise_rng.standard_normal(self.noise_shape)).astype(np.float32)}

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.episode_start = True
        self.state_history = []
        if self.drawer_closer is not None:
            self.drawer_closer.reset()
        print("successfully unset obs and language intruction")
