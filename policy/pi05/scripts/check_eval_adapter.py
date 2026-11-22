"""Checks of deploy_policy.eval() on a mock env (no simulator, no GPU), and optionally of warm_up() on a checkpoint.

1. The adapter sends exactly the same actions and returns the same values as the adapter of the original finals
   submission (commit 26e9a26, read with `git show`), in five cases: success in the middle of a chunk, step limit
   as a tensor and as a plain bool, the simulator-only drawer rule, and a harness that stops without truncation.
2. It counts what the adapter asks of the env per executed action: env.step(), success checks, `env.unwrapped`
   reads and observation reads.
3. `episode_time_limit_s`: with a slow env, eval() returns truncated=True once the budget is used up, and a success
   before that still ends the episode as a success.
4. With --task and --checkpoint: the action chunk sampled after warm_up() equals the one sampled without it.

From policy/pi05:
    .venv/bin/python scripts/check_eval_adapter.py
    .venv/bin/python scripts/check_eval_adapter.py --task item_assembly --checkpoint checkpoints/item_assembly
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument("--task", help="task name, for the warm-up check")
parser.add_argument("--checkpoint", help="the task's checkpoint folder, for the warm-up check")
parser.add_argument("--base", default="26e9a26", help="commit of the adapter to compare with")
args = parser.parse_args()
if not args.checkpoint:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the mock checks need no GPU

PI05 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(PI05))
for path in (REPO, os.path.join(PI05, "src"), os.path.join(PI05, "packages/openpi-client/src"), PI05):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new_dp = load("pi05_deploy_policy_new", os.path.join(PI05, "deploy_policy.py"))
from episode_timing import EpisodeTimer  # noqa: E402

old_dp = None
try:
    source = subprocess.run(
        ["git", "-C", REPO, "show", f"{args.base}:policy/pi05/deploy_policy.py"],
        check=True, capture_output=True, text=True,
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        old_path = os.path.join(tmp, "deploy_policy_base.py")
        with open(old_path, "w") as f:
            f.write(source)
        old_dp = load("pi05_deploy_policy_base", old_path)
except (OSError, subprocess.CalledProcessError) as error:
    print(f"commit {args.base} is not available ({error}); skipping the comparison with it")

COUNTS = {}


def count(key):
    COUNTS[key] = COUNTS.get(key, 0) + 1


class CountingDict(dict):
    def __getitem__(self, key):
        count("observation reads")
        return dict.__getitem__(self, key)


class Box:
    def __init__(self):
        self.shape = (14,)
        self.low = np.full(14, -3.14, dtype=np.float32)
        self.high = np.full(14, 3.14, dtype=np.float32)
        self.low[[6, 13]] = 0.0
        self.high[[6, 13]] = 0.05


class Unwrapped:
    def __init__(self):
        self.single_action_space = Box()
        self.device = torch.device("cpu")


class MockEnv:
    """The parts of the env eval() uses. The observation changes with every step."""

    def __init__(self, success_at=None, limit=1000, plain_bool_truncation=False, step_seconds=0.0):
        self._unwrapped = Unwrapped()
        self.success_at = success_at
        self.limit = limit
        self.plain = plain_bool_truncation
        self.step_seconds = step_seconds
        self.steps = 0
        self.sent = []
        self._current_instruction = "instruction from the env"

    @property
    def unwrapped(self):
        count("env.unwrapped reads")
        return self._unwrapped

    def obs(self):
        k = self.steps
        image = lambda c: np.full((1, 6, 8, 4), (7 * k + c) % 256, dtype=np.uint8)
        return CountingDict(
            sensor=CountingDict(
                cam_high=CountingDict(color=image(1)),
                cam_left_wrist=CountingDict(color=image(2)),
                cam_right_wrist=CountingDict(color=image(3)),
            ),
            robot=CountingDict(qpos=torch.full((1, 14), 0.01 * k)),
        )

    def step(self, action):
        count("env.step calls")
        assert tuple(action.shape) == (1, 14), action.shape
        time.sleep(self.step_seconds)
        self.sent.append(action.detach().cpu().numpy().copy())
        self.steps += 1
        done = self.steps >= self.limit
        truncated = done if self.plain else torch.tensor([done])
        return self.obs(), 0.0, torch.tensor([False]), truncated, {"elapsed_steps": torch.tensor(self.steps)}

    def get_wrapper_attr(self, name):
        assert name == "is_task_success"
        count("success checks")
        return lambda: self.success_at is not None and self.steps >= self.success_at


class FakeModel:
    """The parts of PI0 eval() uses. get_action() depends on the last observation window, so an adapter that
    inferred from a different observation would send different actions."""

    def __init__(self, seed, drawer=False):
        self.rng = np.random.default_rng(seed)
        self.observation_window = None
        self.pytorch_device = "cpu"
        self.pi0_step = 10
        self.rescale_gripper = True
        self.drawer_closer = object() if drawer else None
        self.task_name = "mock"
        self.timer = EpisodeTimer(self.task_name)
        self.env_action_format = None
        self.episode_time_limit_s = None
        self.windows = []

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        self.observation_window = True
        self.windows.append(float(np.asarray(state).sum()) + float(sum(np.asarray(i).sum() for i in img_arr)))

    def get_action(self):
        chunk = self.rng.uniform(-1.2, 1.3, size=(50, 14)).astype(np.float32)
        return chunk + np.float32(1e-3) * np.float32(self.windows[-1] % 97)

    def reset_obsrvationwindows(self):
        self.observation_window = None


def run(dp, env, model, max_steps=1000):
    """The episode loop of scripts/eval_policy.py."""
    obs = env.obs()
    dp.reset_model(model)
    env_steps, returns = 0, []
    while env_steps < max_steps:
        obs, info, truncated, times = dp.eval(env, model, obs)
        env_steps = int(info["elapsed_steps"].item())
        truncated = bool(torch.as_tensor(truncated).any())
        returns.append((env_steps, truncated, len(times)))
        if truncated or env.get_wrapper_attr("is_task_success")():
            break
    return returns


def compare(label, make_env, drawer=False, max_steps=1000):
    results = {}
    for name, dp in (("base", old_dp), ("new", new_dp)):
        if dp is None:
            continue
        COUNTS.clear()
        env, model = make_env(), FakeModel(seed=1234, drawer=drawer)
        results[name] = (env, model, run(dp, env, model, max_steps), dict(COUNTS))
    env, model, returns, counts = results["new"]
    steps = len(env.sent)
    ok = True
    line = f"{label}: {steps} actions in {len(returns)} eval() calls"
    if "base" in results:
        b_env, b_model, b_returns, b_counts = results["base"]
        same_actions = len(b_env.sent) == steps and all(np.array_equal(a, b) for a, b in zip(b_env.sent, env.sent))
        same_windows = b_model.windows == model.windows if drawer else True
        ok = same_actions and b_returns == returns and same_windows
        line += f"; same actions as {args.base}: {same_actions}; same returns: {b_returns == returns}"
        if drawer:
            line += f"; same per-step observations for the drawer rule: {same_windows}"
    print(line)
    for key in ("env.step calls", "success checks", "env.unwrapped reads", "observation reads"):
        base = f"{results['base'][3].get(key, 0) / steps:5.2f} -> " if "base" in results else ""
        print(f"    {key:20s} per action: {base}{counts.get(key, 0) / steps:5.2f}")
    return ok


print("--- eval() on a mock env ([pi05 timing] lines come from the adapter) ---")
ok = compare("success at step 37", lambda: MockEnv(success_at=37))
ok &= compare("step limit at 53 (tensor)", lambda: MockEnv(limit=53))
ok &= compare("step limit at 1000 (plain bool)", lambda: MockEnv(limit=1000, plain_bool_truncation=True))
ok &= compare("drawer rule on, success at step 64", lambda: MockEnv(success_at=64), drawer=True)
ok &= compare("harness stops at 95 steps", lambda: MockEnv(), max_steps=95)

print("--- episode_time_limit_s: 0.3 s budget, env.step takes 20 ms ---")
env, model = MockEnv(step_seconds=0.02), FakeModel(seed=1)
model.episode_time_limit_s = 0.3
returns = run(new_dp, env, model)
limit_ok = returns[-1][1] and 10 <= env.steps <= 25 and not any(r[1] for r in returns[:-1])
print(f"stopped after {env.steps} steps with truncated={returns[-1][1]}: {'ok' if limit_ok else 'WRONG'}")
env, model = MockEnv(success_at=12, step_seconds=0.02), FakeModel(seed=1)
model.episode_time_limit_s = 0.3
returns = run(new_dp, env, model)
success_ok = env.steps == 12 and not returns[-1][1]
print(f"success at step 12 within the budget: {env.steps} steps, truncated={returns[-1][1]}: {'ok' if success_ok else 'WRONG'}")
ok &= limit_ok and success_ok

if args.checkpoint:
    print("--- warm_up() on the checkpoint ---")
    from pi_model import PI0

    pi0 = PI0(task_name=args.task, checkpoint_dir=args.checkpoint, pi0_step=10, pytorch_device="cpu")
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for _ in range(3)]
    pi0.set_language(None)
    pi0.update_observation_window(images, rng.uniform(-0.5, 0.5, size=14).astype(np.float32))
    key = pi0.policy._rng
    started = time.perf_counter()
    without = pi0.get_action()  # the first inference compiles
    first_s = time.perf_counter() - started
    pi0.policy._rng = key  # the same starting key and episode state as for the first chunk
    pi0.episode_start = True
    pi0.warm_up()
    started = time.perf_counter()
    after = pi0.get_action()
    same = np.array_equal(without, after)
    print(f"first inference (compiling) {first_s:.2f} s, inference after warm_up {1000 * (time.perf_counter() - started):.0f} ms")
    print(f"same action chunk with and without warm_up: {same}")
    ok &= same

print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
