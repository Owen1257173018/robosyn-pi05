"""π0.5 policy wrapper used by deploy_policy.py (adapted from policy/pi0/pi_model.py)."""

import dataclasses
import os

import numpy as np
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


class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        pytorch_device="cuda",
        task_name=None,
        checkpoint_dir=None,
        open_grippers_at_start=False,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.pytorch_device = pytorch_device
        self.task_name = task_name
        self.checkpoint_dir = checkpoint_dir or self._default_checkpoint_dir()
        self.open_at_start = np.array(OPEN_AT_START.get(task_name, (False, False)), dtype=bool) & open_grippers_at_start
        self.episode_start = True

        self.policy = self._create_policy(self.pytorch_device)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.instruction = None
        self.pi0_step = pi0_step

    def _default_checkpoint_dir(self):
        return os.path.join(PI05_ROOT, "checkpoints", self.train_config_name, self.model_name, str(self.checkpoint_id))

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

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        actions = np.array(self.policy.infer(self.observation_window)["actions"])
        gripper = actions[:, GRIPPER_DIMS]
        actions[:, GRIPPER_DIMS] = np.where(gripper < GRIPPER_CLOSE_BELOW, 0.0, gripper)
        if self.episode_start and self.open_at_start.any():
            closed = np.asarray(self.observation_window["observation/state"])[GRIPPER_DIMS] < GRIPPER_CLOSE_BELOW
            for dim in np.asarray(GRIPPER_DIMS)[closed & self.open_at_start]:
                actions[:, dim] = 1.0
        self.episode_start = False
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.episode_start = True
        print("successfully unset obs and language intruction")
