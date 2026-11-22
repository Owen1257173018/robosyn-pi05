"""Wall-clock breakdown of each episode, printed as one line when the episode ends.

The organizers asked (finals timeout notice, 2026-11-22) for time to first action, synchronized model inference,
observation/transport, execution/feedback and total episode duration to be reported separately. One line per
episode, e.g.

[pi05 timing] item_assembly episode 3 (success): 412 steps, 42 chunks, 52.31 s total; first action 0.41 s;
model 4.90 s (117 ms/chunk); obs->model 0.02 s; action format 0.01 s; env.step 45.10 s (109 ms/step, max 230 ms);
success check 0.60 s (1.5 ms/step); other in eval 0.10 s; outside eval 1.56 s

- total: from reset_model (the harness calls it right after the env reset) to the end of the last eval() call.
- first action: from reset_model to the first env.step() call.
- model: get_action(), i.e. openpi inference (synchronized: its result is copied to host memory) and the rules.
- obs->model: reading the observation (cameras, qpos) into the model input, once per chunk.
- env.step: the env's own time per action, i.e. execution, feedback and the observation it returns.
- success check: the per-step is_task_success() calls.
- outside eval: time between our eval() calls, spent in the evaluation harness.
"""

import atexit
import time

SLOW_STEP_S = 5.0  # an env.step() slower than this is reported at once


class EpisodeTimer:
    def __init__(self, task_name):
        self.task_name = task_name
        self.episode = 0
        self._clear()
        atexit.register(self.report)

    def _clear(self):
        self.t_start = None
        self.t_first_action = None
        self.t_last = None
        self.steps = 0
        self.chunks = 0
        self.in_eval = 0.0
        self.encode = 0.0
        self.model = 0.0
        self.format = 0.0
        self.step = 0.0
        self.step_max = 0.0
        self.success_check = 0.0
        self.obs_per_step = 0.0
        self.end_reason = None
        self.reported = True
        self.limit_hit = False

    def over_limit(self, limit_s):
        """True once per episode, when `limit_s` seconds have passed since the episode started."""
        if not limit_s or self.limit_hit or time.perf_counter() - self.t_start <= limit_s:
            return False
        self.limit_hit = True
        print(
            f"[pi05 timing] {self.task_name} episode {self.episode}: {limit_s:.0f} s since the reset after "
            f"{self.steps} steps (episode_time_limit_s); ending the episode as a timeout",
            flush=True,
        )
        return True

    def start(self):
        """New episode (reset_model). Reports the previous one if it ended without our eval() seeing the end."""
        self.report()
        self.episode += 1
        self._clear()
        self.t_start = time.perf_counter()
        self.reported = False

    def ensure_started(self):
        if self.t_start is None or self.reported:
            self.start()

    def chunk(self, encode_s, model_s, format_s):
        self.chunks += 1
        self.encode += encode_s
        self.model += model_s
        self.format += format_s

    def before_step(self):
        now = time.perf_counter()
        if self.t_first_action is None:
            self.t_first_action = now
            print(
                f"[pi05 timing] {self.task_name} episode {self.episode}: first action "
                f"{now - self.t_start:.2f} s after the reset",
                flush=True,
            )
        return now

    def after_step(self, step_s, success_s):
        self.steps += 1
        self.step += step_s
        self.step_max = max(self.step_max, step_s)
        self.success_check += success_s
        if step_s > SLOW_STEP_S:
            print(
                f"[pi05 timing] {self.task_name} episode {self.episode}: env.step took {step_s:.2f} s "
                f"(step {self.steps})",
                flush=True,
            )

    def eval_done(self, eval_s, end_reason=None):
        self.in_eval += eval_s
        self.t_last = time.perf_counter()
        if end_reason is not None:
            self.end_reason = end_reason
            self.report()

    def report(self):
        if self.reported or self.t_start is None or self.t_last is None:
            return
        self.reported = True
        total = self.t_last - self.t_start
        first = (self.t_first_action - self.t_start) if self.t_first_action is not None else float("nan")
        other = self.in_eval - (self.encode + self.model + self.format + self.step + self.success_check + self.obs_per_step)
        per_chunk = 1000 * self.model / max(1, self.chunks)
        per_step = 1000 * self.step / max(1, self.steps)
        per_check = 1000 * self.success_check / max(1, self.steps)
        line = (
            f"[pi05 timing] {self.task_name} episode {self.episode} ({self.end_reason or 'ended by the harness'}): "
            f"{self.steps} steps, {self.chunks} chunks, {total:.2f} s total; first action {first:.2f} s; "
            f"model {self.model:.2f} s ({per_chunk:.0f} ms/chunk); obs->model {self.encode:.2f} s; "
            f"action format {self.format:.2f} s; env.step {self.step:.2f} s ({per_step:.0f} ms/step, "
            f"max {1000 * self.step_max:.0f} ms); success check {self.success_check:.2f} s ({per_check:.1f} ms/step); "
        )
        if self.obs_per_step:
            line += f"per-step obs (drawer rule) {self.obs_per_step:.2f} s; "
        line += f"other in eval {other:.2f} s; outside eval {total - self.in_eval:.2f} s"
        print(line, flush=True)
