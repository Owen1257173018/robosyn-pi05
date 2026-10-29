"""Offline action error of a checkpoint on the held-out Real episodes (robosyn_mixture.REAL_HOLDOUT).

Targets go through the same speed-up strides and gripper relabelling as the training mixture of the config,
so the error measures how well the policy reproduces the (sped-up) real demonstrations.

    uv run scripts/robosyn_eval_offline.py --config-name pi05_robosyn_v1_r20 \
        --checkpoint-dir checkpoints/pi05_robosyn_v1_r20/v1_r20/9999 --frame-stride 5
"""

import dataclasses
import json
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.robosyn_mixture as robosyn_mixture

ARM_DIMS = np.array([*range(0, 6), *range(7, 13)])
GRIPPER_CLOSE_BELOW = 0.25  # same snapping as pi_model.PI0.get_action


def main(config_name: str, checkpoint_dir: str, frame_stride: int = 5, horizons: tuple[int, ...] = (10, 25, 50),
         output: str | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    results = {}
    for member in data_config.mixture.members:
        if member.group != "real":
            continue
        dataset = robosyn_mixture.MixtureMemberDataset(
            dataclasses.replace(member, holdout_episodes=()), config.model.action_horizon, 0.01
        )
        frames = np.concatenate(
            [np.arange(dataset.episode_from[e], dataset.episode_to[e], frame_stride) for e in member.holdout_episodes]
        )
        position = {int(f): i for i, f in enumerate(dataset.indices)}
        errors = {h: [] for h in horizons}
        gripper_agree = []
        for frame in tqdm.tqdm(frames, desc=member.repo_id.split("/")[-1]):
            if int(frame) not in position:  # static head/tail frames are not training starts either
                continue
            sample = dataset[position[int(frame)]]
            target = np.asarray(sample.pop("actions"))
            predicted = np.asarray(policy.infer(sample)["actions"])
            for h in horizons:
                errors[h].append(np.abs(predicted[:h, ARM_DIMS] - target[:h, ARM_DIMS]).mean())
            closed_pred = predicted[:10, robosyn_mixture.GRIPPER_DIMS] < GRIPPER_CLOSE_BELOW
            closed_true = target[:10, robosyn_mixture.GRIPPER_DIMS] < GRIPPER_CLOSE_BELOW
            gripper_agree.append((closed_pred == closed_true).mean())
        task = member.repo_id.split("cobotmagic_Real_")[-1]
        results[task] = {f"arm_mae_h{h}": float(np.mean(errors[h])) for h in horizons}
        results[task]["gripper_agreement_h10"] = float(np.mean(gripper_agree))
        results[task]["samples"] = len(gripper_agree)
        print(task, json.dumps(results[task]))

    results["mean"] = {
        key: float(np.mean([r[key] for r in results.values()])) for key in next(iter(results.values())) if key != "samples"
    }
    print("mean", json.dumps(results["mean"]))
    out = pathlib.Path(output or pathlib.Path(checkpoint_dir) / "offline_real_holdout.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"Wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
