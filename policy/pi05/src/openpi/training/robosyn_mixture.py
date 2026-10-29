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
# Arm joints in the 14-dim layout (grippers are dims 6 and 13).
ARM_DIMS = (*range(0, 6), *range(7, 13))

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
    # Drop frames before the arms start moving and after they stop.
    trim_static: bool = False
    holdout_episodes: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class Mixture:
    members: tuple[MixtureMember, ...]
    # (group, share) pairs; shares are normalized.
    group_weights: tuple[tuple[str, float], ...] = (("sim", 0.8), ("real", 0.2))
    # Inside a group, a member's share is proportional to n ** alpha, n = usable start frames (pi0 uses 0.43).
    balance_alpha: float = 0.43
    # A frame counts as moving once an arm joint command leaves the first / last command by this much (rad).
    static_threshold: float = 0.01


def sim_member(task: str) -> MixtureMember:
    if task == "click_bell":  # the only Sim dataset with the older key layout
        return MixtureMember(
            f"RoboSynChallenge/cobotmagic_Sim_{task}", "sim", state_key="observation.qpos", image_key="{camera}.color"
        )
    return MixtureMember(f"RoboSynChallenge/cobotmagic_Sim_{task}", "sim")


def real_member(task: str, action_stride: float = 1.0) -> MixtureMember:
    return MixtureMember(
        f"RoboSynChallenge/cobotmagic_Real_{task}",
        "real",
        dim_indices=REAL_TO_14,
        action_stride=action_stride,
        trim_static=True,
        holdout_episodes=REAL_HOLDOUT,
    )


def robosyn_mixture(
    real_share: float = 0.2,
    real_strides: tuple[tuple[str, float], ...] = (),
    tasks: tuple[str, ...] = TASKS,
) -> Mixture:
    strides = dict(real_strides)
    members = tuple(m for task in tasks for m in (sim_member(task), real_member(task, strides.get(task, 1.0))))
    return Mixture(members=members, group_weights=(("sim", 1.0 - real_share), ("real", real_share)))


def action_offsets(stride: float, horizon: int) -> np.ndarray:
    return np.array([round(t * stride) for t in range(horizon)], dtype=np.int64)


def static_head_tail(arm_actions: np.ndarray, threshold: float) -> tuple[int, int]:
    """Frames before the first and after the last arm movement of one episode."""
    moved_from_start = np.nonzero(np.abs(arm_actions - arm_actions[0]).max(axis=1) > threshold)[0]
    moved_from_end = np.nonzero(np.abs(arm_actions - arm_actions[-1]).max(axis=1) > threshold)[0]
    if len(moved_from_start) == 0:
        return 0, 0
    return int(moved_from_start[0]), int(len(arm_actions) - 1 - moved_from_end[-1])


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
        self.episode_from = self._dataset.episode_data_index["from"].numpy()
        self.episode_to = self._dataset.episode_data_index["to"].numpy()
        self.indices = self._start_indices(static_threshold)

    def load_columns(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(state, action, episode_index) of every stored frame, without decoding videos."""
        columns = [self.member.state_key, "action", "episode_index"]
        table = self._dataset.hf_dataset.with_format("numpy", columns=columns)[:]
        state = np.stack(table[self.member.state_key]).astype(np.float32)
        action = np.stack(table["action"]).astype(np.float32)
        if self._dims is not None:
            state, action = state[:, self._dims], action[:, self._dims]
        return state, action, np.asarray(table["episode_index"])

    def _start_indices(self, static_threshold: float) -> np.ndarray:
        _, action, episode_index = self.load_columns()
        if not np.array_equal(episode_index[self.episode_from], np.arange(len(self.episode_from))):
            raise ValueError(f"{self.member.repo_id}: episodes are not stored as 0..N-1 in order")
        holdout = set(self.member.holdout_episodes)
        starts = []
        for episode, (lo, hi) in enumerate(zip(self.episode_from, self.episode_to, strict=True)):
            if episode in holdout:
                continue
            if self.member.trim_static:
                head, tail = static_head_tail(action[lo:hi][:, ARM_DIMS], static_threshold)
                lo, hi = lo + head, hi - tail
            starts.append(np.arange(lo, hi))
        return np.concatenate(starts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index) -> dict:
        item = self._dataset[int(self.indices[index])]
        state = np.asarray(item[self.member.state_key], dtype=np.float32)
        actions = np.asarray(item["action"], dtype=np.float32)
        if self._dims is not None:
            state, actions = state[..., self._dims], actions[..., self._dims]
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
