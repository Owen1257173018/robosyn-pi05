"""Weighted sim + real LeRobot mixtures for RoboSynChallenge co-training.

The official datasets do not share one schema: Real stores a 32-dim state/action (per arm 6 joints, gripper,
ee pose) under `observation.images.*`; Sim click_bell stores `observation.qpos` + `cam_*.color`; the other Sim
datasets store a 14-dim `observation.state` + `observation.images.*`. Each `MixtureMember` maps its dataset to
the keys `EmbodiChainInputs` expects, so the downstream transforms and the inference path stay unchanged.

Held-out episodes and static frames are removed from the sampler's start indices instead of through
`LeRobotDataset(episodes=...)`: this LeRobot version looks episode bounds up by position, which is only
correct when the full dataset is loaded.
"""

import dataclasses
import logging

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

TASKS = (
    "click_bell",
    "handle_basket",
    "water_pouring",
    "table_rearrangement",
    "items_handover",
    "drawer_open_place",
    "mixer_operating",
    "item_assembly",
    "manipulate_pipette",
    "sample_loading",
)
# Real state/action: per arm [6 joints, gripper, ee_pos x3, ee_rot6d x6] -> per arm [6 joints, gripper].
REAL_TO_14 = (*range(0, 7), *range(16, 23))
# 6 of the 60 episodes of every Real dataset are kept for offline validation.
REAL_HOLDOUT = (0, 10, 20, 30, 40, 50)
# Grippers in the 14-dim layout (the other dims are the 6 joints of each arm).
GRIPPER_DIMS = np.array([6, 13])

_CAMERAS = {
    "observation/image": "cam_high",
    "observation/left_wrist_image": "cam_left_wrist",
    "observation/right_wrist_image": "cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class MixtureMember:
    """One LeRobot dataset in a training mixture."""

    repo_id: str
    # Sampling group ("sim" / "real"); each group gets a fixed share of the samples.
    group: str
    state_key: str = "observation.state"
    # Video key pattern, `{camera}` is cam_high / cam_left_wrist / cam_right_wrist.
    image_key: str = "observation.images.{camera}"
    # Indices of the 14 policy dims in the stored state/action; None keeps the stored vector.
    dim_indices: tuple[int, ...] | None = None
    # Stored frames advanced per action step. >1 replays the demonstration faster; every frame stays a valid
    # start, so all phases of the subsampling are used.
    action_stride: float = 1.0
    # Drop frames before the robot starts moving and after it stops.
    trim_static: bool = False
    holdout_episodes: tuple[int, ...] = ()
    # Gripper actions are trained as the normalized opening (0 closed, 1 fully open), like the state.
    # Sim stores the raw finger target in meters, which the env clips to the 0.05 m joint limit (most experts
    # command 1.0 to open); scaling and clipping maps it to the normalized opening.
    gripper_action_scale: float | None = None
    # Real stores the measured opening as the action, which stays at the object width while holding; label
    # closing and holding as 0 (close), as the teleop leader commanded.
    relabel_gripper_hold: bool = False
    # Frames the measured opening rests (within 0.02) before a closing / opening motion; with it set, the motion
    # is labelled from the end of that rest instead of from the last frame near the previous extreme, so a
    # partial close that pauses before the grasp is not labelled closed. None keeps the v1 labels.
    gripper_rest_frames: int | None = None
    # Probability of replacing each gripper's state input with a random opening. The v1 policy copies the gripper
    # state, so it never opens grippers the sim env resets closed (the demonstrations open them within a frame or
    # start open); with noise the opening has to come from the images.
    gripper_state_noise: float = 0.0


@dataclasses.dataclass(frozen=True)
class Mixture:
    members: tuple[MixtureMember, ...]
    # (group, share) pairs; shares are normalized.
    group_weights: tuple[tuple[str, float], ...] = (("sim", 0.8), ("real", 0.2))
    # Inside a group, a member's share is proportional to n ** alpha, n = usable start frames (pi0 uses 0.43).
    balance_alpha: float = 0.43
    # A frame counts as moving once an action dim leaves the first / last action by this much.
    static_threshold: float = 0.01


# Action strides that bring a dataset's moving segments under the step limit H of the random setting
# (real: p90 moving length <= 0.75 H, capped at 2x; sim items_handover: the expert alone takes 327 of 350 steps).
# The real robot has had a 1000-step limit for every task since FAQ2-18 (2026-10-30), which every Real
# moving segment already meets, so the v3 configs train Real at its recorded speed.
REAL_STRIDES = (
    ("handle_basket", 1.2),
    ("water_pouring", 1.9),
    ("table_rearrangement", 1.7),
    ("items_handover", 1.8),
    ("mixer_operating", 1.3),
    ("item_assembly", 1.8),
    ("sample_loading", 1.6),
)
SIM_STRIDES = (("items_handover", 1.2),)


def sim_member(
    task: str, action_stride: float = 1.0, *, trim_static: bool = False, gripper_state_noise: float = 0.0
) -> MixtureMember:
    # drawer_open_place already stores normalized gripper actions.
    scale = None if task == "drawer_open_place" else 1 / 0.05
    keys = {}
    if task == "click_bell":  # the only Sim dataset with the older key layout
        keys = {"state_key": "observation.qpos", "image_key": "{camera}.color"}
    return MixtureMember(
        f"RoboSynChallenge/cobotmagic_Sim_{task}",
        "sim",
        action_stride=action_stride,
        trim_static=trim_static,
        gripper_action_scale=scale,
        gripper_state_noise=gripper_state_noise,
        **keys,
    )


def gen_repo_id(task: str, suffix: str = "") -> str:
    """Sim episodes generated locally with the task's expert (WORKLOG P3.50), stored like the official Sim data.

    `suffix` names a later generation with a changed expert, e.g. "2" for item_assembly's fixed docking (WORKLOG P3.80).
    """
    return f"RoboSynGen/cobotmagic_Sim_{task}_gen{suffix}"


def real_member(
    task: str,
    action_stride: float = 1.0,
    *,
    gripper_rest_frames: int | None = None,
    gripper_state_noise: float = 0.0,
    holdout: bool = True,
) -> MixtureMember:
    return MixtureMember(
        f"RoboSynChallenge/cobotmagic_Real_{task}",
        "real",
        dim_indices=REAL_TO_14,
        action_stride=action_stride,
        trim_static=True,
        holdout_episodes=REAL_HOLDOUT if holdout else (),
        relabel_gripper_hold=True,
        gripper_rest_frames=gripper_rest_frames,
        gripper_state_noise=gripper_state_noise,
    )


def robosyn_mixture(
    real_share: float = 0.2,
    real_strides: tuple[tuple[str, float], ...] = (),
    sim_strides: tuple[tuple[str, float], ...] = (),
    tasks: tuple[str, ...] = TASKS,
    *,
    trim_sim: bool = False,
    gripper_rest_frames: int | None = None,
    gripper_state_noise: float = 0.0,
    real_holdout: bool = True,
    extra_sim: tuple[tuple[str, str], ...] = (),
) -> Mixture:
    """`extra_sim`: (task, repo_id) pairs of more Sim datasets of a task, read like the task's official one."""
    real, sim = dict(real_strides), dict(sim_strides)
    members = tuple(
        m
        for task in tasks
        for m in (
            sim_member(task, sim.get(task, 1.0), trim_static=trim_sim, gripper_state_noise=gripper_state_noise),
            *(
                # Recorded with the current EmbodiChain, so always the default key layout, also for click_bell.
                dataclasses.replace(
                    sim_member(task, sim.get(task, 1.0), trim_static=trim_sim, gripper_state_noise=gripper_state_noise),
                    repo_id=repo_id,
                    state_key=MixtureMember.state_key,
                    image_key=MixtureMember.image_key,
                )
                for extra_task, repo_id in extra_sim
                if extra_task == task
            ),
            real_member(
                task,
                real.get(task, 1.0),
                gripper_rest_frames=gripper_rest_frames,
                gripper_state_noise=gripper_state_noise,
                holdout=real_holdout,
            ),
        )
    )
    return Mixture(members=members, group_weights=(("sim", 1.0 - real_share), ("real", real_share)))


def action_offsets(stride: float, horizon: int) -> np.ndarray:
    return np.array([round(t * stride) for t in range(horizon)], dtype=np.int64)


def static_head_tail(actions: np.ndarray, threshold: float) -> tuple[int, int]:
    """Frames before the first and after the last movement of one episode."""
    moved_from_start = np.nonzero(np.abs(actions - actions[0]).max(axis=1) > threshold)[0]
    moved_from_end = np.nonzero(np.abs(actions - actions[-1]).max(axis=1) > threshold)[0]
    if len(moved_from_start) == 0:
        return 0, 0
    return int(moved_from_start[0]), int(len(actions) - 1 - moved_from_end[-1])


def _motion_start(opening: np.ndarray, last: int, t: int, settle: float, rest: int) -> int:
    """First frame of the motion that switched the mode at `t`: right after the latest `rest` frames (all after
    `last`) that stay within `settle` of each other, or right after `last` if the gripper never rested."""
    for k in range(t, last + rest, -1):
        window = opening[k - rest : k]
        if window.max() - window.min() <= settle:
            return k
    return last + 1


def relabel_gripper_hold(
    opening: np.ndarray, delta: float = 0.08, settle: float = 0.02, rest: int | None = None
) -> np.ndarray:
    """Command labels for one episode's measured gripper opening: 0 from the start of a closing motion until
    the gripper opens again. Moving more than `delta` past the running max (open) / min (closed) switches the
    mode; the motion starts after the last frame within `settle` of that extreme, so a hold that settles
    slightly above the closing overshoot stays labelled closed. With `rest`, the motion starts after the last
    rest of `rest` frames instead (see `MixtureMember.gripper_rest_frames`)."""
    labels = opening.astype(np.float32).copy()
    closed = bool(opening[0] < delta)
    extreme, last = opening[0], 0
    for t, value in enumerate(opening):
        if closed:
            if value - extreme > delta:
                closed = False
                start = _motion_start(opening, last, t, settle, rest) if rest else last + 1
                labels[start : t + 1] = opening[start : t + 1]
                extreme, last = value, t
            elif value <= extreme + settle:
                extreme, last = min(extreme, value), t
        elif extreme - value > delta:
            closed = True
            labels[(_motion_start(opening, last, t, settle, rest) if rest else last + 1) : t + 1] = 0.0
            extreme, last = value, t
        elif value >= extreme - settle:
            extreme, last = max(extreme, value), t
        if closed:
            labels[t] = 0.0
    return labels


class MixtureMemberDataset(torch.utils.data.Dataset):
    def __init__(self, member: MixtureMember, action_horizon: int, static_threshold: float):
        self.member = member
        meta = lerobot_dataset.LeRobotDatasetMetadata(member.repo_id)
        self.offsets = action_offsets(member.action_stride, action_horizon)
        self._dataset = lerobot_dataset.LeRobotDataset(
            member.repo_id, delta_timestamps={"action": [int(o) / meta.fps for o in self.offsets]}
        )
        self._tasks = dict(meta.tasks)
        self._dims = None if member.dim_indices is None else np.asarray(member.dim_indices)
        # Copies, not `.numpy()` views: pickling the dataset for spawn workers moves the tensors to shared memory,
        # and a view of the old storage would then read freed memory.
        self.episode_from = np.array(self._dataset.episode_data_index["from"])
        self.episode_to = np.array(self._dataset.episode_data_index["to"])
        state, action, episode_index = self._read_columns()
        if not np.array_equal(episode_index[self.episode_from], np.arange(len(self.episode_from))):
            raise ValueError(f"{member.repo_id}: episodes are not stored as 0..N-1 in order")
        self._gripper_labels = None
        if member.relabel_gripper_hold:
            self._gripper_labels = np.empty((len(action), len(GRIPPER_DIMS)), dtype=np.float32)
            for lo, hi in zip(self.episode_from, self.episode_to, strict=True):
                for i, dim in enumerate(GRIPPER_DIMS):
                    self._gripper_labels[lo:hi, i] = relabel_gripper_hold(
                        action[lo:hi, dim], rest=member.gripper_rest_frames
                    )
        self.indices = self._start_indices(self._fix_gripper(action), static_threshold)

    def _read_columns(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        columns = [self.member.state_key, "action", "episode_index"]
        table = self._dataset.hf_dataset.with_format("numpy", columns=columns)[:]
        state = np.stack(table[self.member.state_key]).astype(np.float32)
        action = np.stack(table["action"]).astype(np.float32)
        if self._dims is not None:
            state, action = state[:, self._dims], action[:, self._dims]
        return state, action, np.asarray(table["episode_index"])

    def _fix_gripper(self, action: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        """Gripper actions in the training convention; `frames` are the stored frame indices of `action`'s rows
        (default: all frames)."""
        if self.member.gripper_action_scale is not None:
            action[:, GRIPPER_DIMS] = np.clip(action[:, GRIPPER_DIMS] * self.member.gripper_action_scale, 0.0, 1.0)
        if self._gripper_labels is not None:
            action[:, GRIPPER_DIMS] = self._gripper_labels if frames is None else self._gripper_labels[frames]
        return action

    def load_columns(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(state, action, episode_index) of every stored frame, without decoding videos."""
        state, action, episode_index = self._read_columns()
        return state, self._fix_gripper(action), episode_index

    def _start_indices(self, action: np.ndarray, static_threshold: float) -> np.ndarray:
        holdout = set(self.member.holdout_episodes)
        starts = []
        for episode, (lo, hi) in enumerate(zip(self.episode_from, self.episode_to, strict=True)):
            if episode in holdout:
                continue
            if self.member.trim_static:
                head, tail = static_head_tail(action[lo:hi], static_threshold)
                lo, hi = lo + head, hi - tail
            starts.append(np.arange(lo, hi))
        return np.concatenate(starts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> dict:
        start = int(self.indices[index])
        item = self._dataset[start]
        state = np.asarray(item[self.member.state_key], dtype=np.float32)
        actions = np.asarray(item["action"], dtype=np.float32)
        if self._dims is not None:
            state, actions = state[..., self._dims], actions[..., self._dims]
        # Frames of the action chunk, clamped to the episode's last frame like LeRobot does.
        episode_end = self.episode_to[np.searchsorted(self.episode_to, start, side="right")]
        actions = self._fix_gripper(actions, np.minimum(start + self.offsets, episode_end - 1))
        if self.member.gripper_state_noise:
            # torch's RNG, which the DataLoader seeds per worker.
            noisy = GRIPPER_DIMS[(torch.rand(len(GRIPPER_DIMS)) < self.member.gripper_state_noise).numpy()]
            state = state.copy()
            state[noisy] = torch.rand(len(noisy)).numpy()
        sample = {key: item[self.member.image_key.format(camera=camera)] for key, camera in _CAMERAS.items()}
        sample["observation/state"] = state
        sample["actions"] = actions
        sample["prompt"] = self._tasks[int(item["task_index"])]
        return sample


def member_weights(mixture: Mixture, sizes: list[int]) -> np.ndarray:
    """Share of the samples drawn from each member."""
    group_weights = dict(mixture.group_weights)
    total = sum(group_weights.values())
    balanced = np.asarray(sizes, dtype=np.float64) ** mixture.balance_alpha
    weights = []
    for member, size_term in zip(mixture.members, balanced, strict=True):
        group_total = sum(b for m, b in zip(mixture.members, balanced, strict=True) if m.group == member.group)
        weights.append(group_weights[member.group] / total * size_term / group_total)
    return np.asarray(weights)


def build_mixture(mixture: Mixture, action_horizon: int) -> tuple[torch.utils.data.ConcatDataset, np.ndarray]:
    """The concatenated member datasets and the per-sample sampling weights."""
    members = [MixtureMemberDataset(m, action_horizon, mixture.static_threshold) for m in mixture.members]
    weights = member_weights(mixture, [len(m) for m in members])
    for member, weight in zip(members, weights, strict=True):
        logging.info(
            f"mixture: {member.member.repo_id} frames={len(member)} share={weight:.4f} "
            f"stride={member.member.action_stride}"
        )
    sample_weights = np.concatenate([np.full(len(m), w / len(m)) for m, w in zip(members, weights, strict=True)])
    return torch.utils.data.ConcatDataset(members), sample_weights


def sample_state_actions(
    mixture: Mixture, action_horizon: int, num_samples: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """States [N, 14] and action chunks [N, H, 14] drawn like the training sampler, without decoding videos."""
    rng = np.random.default_rng(seed)
    members = [MixtureMemberDataset(m, action_horizon, mixture.static_threshold) for m in mixture.members]
    counts = rng.multinomial(num_samples, member_weights(mixture, [len(m) for m in members]))
    states, chunks = [], []
    for member, count in zip(members, counts, strict=True):
        state, action, episode_index = member.load_columns()
        starts = rng.choice(member.indices, size=count, replace=True)
        episode_end = member.episode_to[episode_index[starts]]
        # LeRobot clamps chunk indices to the episode's last frame.
        frames = np.minimum(starts[:, None] + member.offsets[None, :], episode_end[:, None] - 1)
        states.append(state[starts])
        chunks.append(action[frames])
    return np.concatenate(states), np.concatenate(chunks)
