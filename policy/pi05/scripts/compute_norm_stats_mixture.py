"""Normalization stats for a RoboSynChallenge mixture config (robosyn_mixture.py), without decoding videos.

Draws (state, action chunk) samples the way the training sampler does, applies the config's data transforms
(EmbodiChainInputs + DeltaActions) and accumulates openpi's RunningStats, i.e. the same statistics
`scripts/compute_norm_stats.py` computes, on a weighted sample of the mixture.

    uv run scripts/compute_norm_stats_mixture.py --config-name pi05_robosyn_mix_r20
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.robosyn_mixture as robosyn_mixture
import openpi.transforms as transforms


def main(config_name: str, num_samples: int = 1_000_000, batch_size: int = 10_000, seed: int = 0):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.mixture is None:
        raise ValueError(f"{config_name} has no mixture; use scripts/compute_norm_stats.py")

    states, actions = robosyn_mixture.sample_state_actions(
        data_config.mixture, config.model.action_horizon, num_samples, seed
    )
    # Shuffled, so the first batch (which sets RunningStats' histogram range) represents the whole mixture.
    order = np.random.default_rng(seed).permutation(len(states))
    data_transform = transforms.compose(data_config.data_transforms.inputs)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for start in tqdm.tqdm(range(0, len(order), batch_size), desc="Computing stats"):
        batch = order[start : start + batch_size]
        sample = data_transform(
            {
                "observation/image": image,
                "observation/left_wrist_image": image,
                "observation/right_wrist_image": image,
                "observation/state": states[batch],
                "actions": actions[batch],  # fancy indexing copies, DeltaActions edits in place
                "prompt": "",
            }
        )
        for key, key_stats in stats.items():
            key_stats.update(np.asarray(sample[key]))

    norm_stats = {key: key_stats.get_statistics() for key, key_stats in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
