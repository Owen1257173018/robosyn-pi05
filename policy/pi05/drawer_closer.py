"""Close the drawer after the object is placed (drawer_open_place).

FAQ2-10 counts drawer_open_place as a success only once the drawer is closed again, but none of the 60 official
Real demonstrations closes it: the right hand grips the handle, pulls the drawer open and lets go, the left hand
puts the tomato in, and the episode ends. The Sim demonstrations close it while still holding the handle.

After the left gripper has put down what it held, this rule replays the right arm's pull in reverse: back to the
pose where it let go of the handle, grip, push to the pose where it first gripped the handle (drawer closed),
let go and stay there. It reads only the proprioception history (joint positions and gripper openings, FAQ2-20).
"""

import numpy as np

LEFT_GRIPPER, RIGHT_ARM, RIGHT_GRIPPER = 6, slice(7, 13), 13


def closed_spans(opening, drop=0.15, min_len=5):
    """(start, end) of the spans where a gripper is closed: its opening fell `drop` below the running maximum
    (holding an object stops it at the object's width), until it opens `drop` above the span's minimum again.
    A span still closed at the end of `opening` ends at len(opening)."""
    spans, start, peak = [], None, opening[0]
    for t, value in enumerate(opening):
        if start is None:
            peak = max(peak, value)
            if value < peak - drop:
                start = t
        elif value > opening[start : t + 1].min() + drop:
            if t - start >= min_len:
                spans.append((start, t))
            start, peak = None, value
    if start is not None and len(opening) - start >= min_len:
        spans.append((start, len(opening)))
    return spans


def _move(q_from, q_to, gripper, max_joint_step, min_steps=5):
    """Right-arm rows (6 joints + gripper) moving linearly in joint space, at most max_joint_step rad per step."""
    steps = max(min_steps, int(np.ceil(np.abs(q_to - q_from).max() / max_joint_step)))
    joints = q_from + (q_to - q_from) * np.linspace(1.0 / steps, 1.0, steps)[:, None]
    return np.concatenate([joints, np.full((steps, 1), gripper)], axis=1)


def _hold(q, gripper, steps):
    return np.tile(np.append(q, gripper), (steps, 1))


class DrawerCloser:
    """Overrides the right arm (dims 7-13 of the 14-dim action) once the closing motion is planned."""

    def __init__(self, min_hold=30, min_hold_opening=0.1, wait_after_release=10, max_joint_step=0.02,
                 grip_steps=8, overshoot=0.05):
        self.min_hold = min_hold  # steps the left gripper must have held something
        self.min_hold_opening = min_hold_opening  # an empty grasp closes to ~0; the tomato keeps it above 0.17
        self.wait_after_release = wait_after_release  # steps for the left hand to leave the drawer
        self.max_joint_step = max_joint_step
        self.grip_steps = grip_steps
        self.overshoot = overshoot  # push this fraction of the pull further, so the drawer closes fully
        self.reset()

    def reset(self):
        self.plan = None  # remaining right-arm rows; the last row is held after the plan ends
        self.triggered_at = None

    def _make_plan(self, history):
        holds = [
            (a, b)
            for a, b in closed_spans(history[:, LEFT_GRIPPER])
            if b < len(history) and b - a >= self.min_hold and history[a:b, LEFT_GRIPPER].min() >= self.min_hold_opening
        ]
        if not holds or len(history) - holds[-1][1] < self.wait_after_release:
            return None
        release = holds[-1][1]
        pulls = [(a, b) for a, b in closed_spans(history[:, RIGHT_GRIPPER]) if a < release]
        if not pulls:
            return None
        grip_start, grip_end = pulls[-1]
        q_now = history[-1, RIGHT_ARM]
        q_closed = history[grip_start, RIGHT_ARM]  # gripped the handle, drawer still closed
        if grip_end == len(history):  # still holding the handle (the Sim way): push straight back
            q_open = q_now
            rows = []
        else:  # let go after pulling (the Real way): go back to the handle and grip it again
            q_open = history[grip_end - 1, RIGHT_ARM]
            rows = [_move(q_now, q_open, 1.0, self.max_joint_step), _hold(q_open, 0.0, self.grip_steps)]
        q_push = q_closed + self.overshoot * (q_closed - q_open)
        rows += [_move(q_open, q_push, 0.0, self.max_joint_step), _hold(q_push, 1.0, self.grip_steps)]
        return np.concatenate(rows)

    def __call__(self, actions, history, executed):
        """actions: (horizon, 14) model actions; history: (T, 14) states so far; executed: actions run per call."""
        if self.plan is None and self.triggered_at is None:
            self.plan = self._make_plan(history)
            if self.plan is None:
                return actions
            self.triggered_at = len(history)
            print(f"[drawer_closer] closing the drawer from step {len(history)}: {len(self.plan)} steps")
        actions = np.array(actions, copy=True)
        rows = np.concatenate([self.plan, np.repeat(self.plan[-1:], max(0, len(actions) - len(self.plan)), axis=0)])
        actions[:, 7:14] = rows[: len(actions)]
        self.plan = self.plan[executed:] if len(self.plan) > executed else self.plan[-1:]
        return actions
