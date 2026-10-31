"""Close the drawer after the object is placed (drawer_open_place).

The simulator counts drawer_open_place as a success only when the object ends within 0.1 m of the drawer body in
the XY plane, which needs the drawer closed again, and the real robot is set up like the simulator (FAQ2-11). But
none of the 60 official Real demonstrations closes it: the right hand grips the handle, pulls the drawer open and
lets go, the left hand puts the tomato in, and the episode ends. The Sim demonstrations close it while still
holding the handle.

After the left gripper has carried something and let go of it, the right arm retraces its own recorded path:
back to where it let go of the handle, grip, back along the pull to where its fingers closed on the handle
(drawer closed) and a little further, let go and stay there. It reads only the proprioception history (joint
positions and gripper openings, FAQ2-17) and the gripper commands the policy has sent.
"""

import numpy as np

LEFT_ARM, LEFT_GRIPPER, RIGHT_ARM, RIGHT_GRIPPER = slice(0, 6), 6, slice(7, 13), 13


def closed_spans(opening, drop=0.15, min_len=5, persist=1):
    """(start, end, open_level) of the spans where a gripper is closed: its opening fell `drop` below the running
    maximum `open_level` (holding an object stops it at the object's width), until it is `drop` above the span's
    minimum again for `persist` steps in a row. A span still closed at the end of `opening` ends at len(opening)."""
    spans, start, peak, low, above = [], None, (opening[0] if len(opening) else 0.0), 0.0, 0
    for t, value in enumerate(opening):
        if start is None:
            peak = max(peak, value)
            if value < peak - drop:
                start, low, above = t, value, 0
            continue
        low = min(low, value)
        above = above + 1 if value > low + drop else 0
        if above >= persist:
            end = t - persist + 1
            if end - start >= min_len:
                spans.append((start, end, peak))
            start, peak = None, value
    if start is not None and len(opening) - start >= min_len:
        spans.append((start, len(opening), peak))
    return spans


def handle_poses(command, opening, release, max_grip_ratio=0.3, persist=5):
    """Steps of the right gripper's handle grasp: (t_closed, t_open), or None if it never gripped the handle before
    `release` (the step the left hand let go of the object).

    command: the right gripper commands sent so far, one per step; opening: its measured opening at those steps.
    The grips are found on the commands: in the simulator the handle pries the fingers apart during the pull while
    they are still commanded shut, and the first few actions of a chunk can briefly copy that opening (`persist`).
    Gripping the thin handle shuts the gripper almost completely (Real data: at most 0.16 of the open level), while
    the operators sometimes squeeze it halfway on the way to the handle (0.44 and up), which is not a grip.
    t_closed: the fingers closed on the handle with the drawer still shut (first grip that started before the
    release); the span itself starts earlier when the gripper half-closes on the way to the handle.
    t_open: the last step of the latest grip (where it let go of the drawer), or None if it still holds it.
    """
    grips = [(a, b) for a, b, level in closed_spans(command, persist=persist)
             if command[a:b].min() <= max_grip_ratio * level]
    pulls = [(a, b) for a, b in grips if a < release]
    if not pulls:
        return None
    a, b = pulls[0]
    t_closed = a + int(np.argmax(opening[a:b] <= opening[a:b].min() + 0.02))
    last_end = grips[-1][1]
    return t_closed, (None if last_end == len(command) else last_end - 1)


def _thin(path, eps=0.01):
    """Points of a joint path at least eps rad apart (largest joint change), both ends kept: drops pauses and
    encoder jitter before the path is replayed."""
    kept = [path[0]]
    for q in path[1:-1]:
        if np.abs(q - kept[-1]).max() >= eps:
            kept.append(q)
    return np.array(kept + [path[-1]])


def _arc(points):
    """Cumulative length of a joint polyline, measured by the largest joint change of each segment."""
    return np.concatenate([[0.0], np.cumsum(np.abs(np.diff(points, axis=0)).max(axis=1))])


def _follow(points, gripper, max_joint_step, min_steps=5):
    """Right-arm rows (6 joints + gripper) along the joint polyline `points`, at most max_joint_step rad per step."""
    s = _arc(points)
    steps = max(min_steps, int(np.ceil(s[-1] / max_joint_step)))
    at = np.linspace(s[-1] / steps, s[-1], steps)
    joints = np.stack([np.interp(at, s, points[:, j]) for j in range(points.shape[1])], axis=1)
    return np.concatenate([joints, np.full((steps, 1), gripper)], axis=1)


def _hold(q, gripper, steps):
    return np.tile(np.append(q, gripper), (steps, 1))


class DrawerCloser:
    """Overrides the right arm (dims 7-13 of the 14-dim action) once the closing motion is planned."""

    def __init__(self, min_hold=30, min_hold_opening=0.1, min_carry=0.5, wait_after_release=10, max_joint_step=0.02,
                 grip_steps=8, overshoot=0.05):
        self.min_hold = min_hold  # steps the left gripper must have held something
        self.min_hold_opening = min_hold_opening  # an empty grasp closes to ~0; the tomato keeps it above 0.17
        # joint-space distance (rad) the left arm must have moved between grasp and release: the Real demonstrations
        # carry the tomato 0.83 or more, a half-closed hover over it in the simulator moved 0.33
        self.min_carry = min_carry
        self.wait_after_release = wait_after_release  # steps for the left hand to leave the drawer
        self.max_joint_step = max_joint_step
        self.grip_steps = grip_steps
        self.overshoot = overshoot  # push this fraction of the pull further, so the drawer closes fully
        self.reset()

    def reset(self):
        self.plan = None  # remaining right-arm rows; the last row is held after the plan ends
        self.triggered_at = None
        self.commands = []  # right gripper commands sent, one per executed step

    def release_step(self, history):
        """Step at which the left gripper let go of the last thing it carried, or None."""
        holds = [
            (a, b)
            for a, b, _ in closed_spans(history[:, LEFT_GRIPPER])
            if b < len(history) and b - a >= self.min_hold and history[a:b, LEFT_GRIPPER].min() >= self.min_hold_opening
            and np.linalg.norm(history[b - 1, LEFT_ARM] - history[a, LEFT_ARM]) >= self.min_carry
        ]
        return holds[-1][1] if holds else None

    def _make_plan(self, history):
        release = self.release_step(history)
        if release is None or len(history) - release < self.wait_after_release:
            return None
        command = np.concatenate(self.commands)[: len(history)]
        poses = handle_poses(command, history[: len(command), RIGHT_GRIPPER], release)
        if poses is None:
            return None
        t_closed, t_open = poses
        # The arm retraces its own recorded joint path: the Real pulls curve in joint space (median 16% of their
        # length off the straight line between their ends), and the path it took while holding the handle is the
        # one the drawer allows. pull[0] is where it gripped the handle with the drawer still closed.
        arm = history[:, RIGHT_ARM]
        rows = []
        if t_open is None:  # still holding the handle (the Sim way): push back from here
            pull = _thin(arm[t_closed:])
        else:  # let go after pulling (the Real way): back along its path to the handle, open, then grip it again
            pull = _thin(arm[t_closed : t_open + 1])
            back = _thin(arm[t_open:])[::-1]
            rows += [_follow(back, 1.0, self.max_joint_step), _hold(arm[t_open], 0.0, self.grip_steps)]
        # then a little past the closed pose, continuing the way the pull started (its first fifth)
        s = _arc(pull)
        lead = pull[0] - pull[min(int(np.argmax(s >= 0.2 * s[-1])), len(pull) - 1)]
        q_push = pull[0] + self.overshoot * s[-1] * lead / max(np.abs(lead).max(), 1e-9)
        push = _follow(np.vstack([pull[::-1], q_push]), 0.0, self.max_joint_step)
        rows += [push, _hold(q_push, 1.0, self.grip_steps)]
        return np.concatenate(rows)

    def __call__(self, actions, history, executed):
        """actions: (horizon, 14) model actions; history: (T, 14) states so far; executed: actions run per call."""
        if self.plan is None and self.triggered_at is None:
            self.plan = self._make_plan(history) if self.commands else None
            if self.plan is None:
                self.commands.append(np.asarray(actions)[:executed, RIGHT_GRIPPER])
                return actions
            self.triggered_at = len(history)
            print(f"[drawer_closer] closing the drawer from step {len(history)}: {len(self.plan)} steps")
        actions = np.array(actions, copy=True)
        rows = np.concatenate([self.plan, np.repeat(self.plan[-1:], max(0, len(actions) - len(self.plan)), axis=0)])
        actions[:, 7:14] = rows[: len(actions)]
        self.plan = self.plan[executed:] if len(self.plan) > executed else self.plan[-1:]
        return actions
