###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""RMA phase-2 loop: dataset collection, student supervised training, deploy wrapper, eval.

Input:  a trained RMA teacher checkpoint (from ``train_loop.run`` on ``Mjlab-G1-Reach-RMA``).
Output: student .pt (AdaptationModule weights) + eval metrics.

Phase 1 (teacher) reuses ``src.train_loop.run`` unchanged — the RMA task registers
privileged mass as an obs term visible to both actor and critic.

Slice constants below MUST match the observation term order in ``config/policy_reach_rma.py``.
If you add/remove obs terms, update these.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Slice indices into the concatenated actor obs from policy_reach_rma.py.
PROPRIO_SLICE = slice(9, 96)
MASS_SLICE    = slice(108, 110)
PROPRIO_DIM   = 87
MASS_DIM      = 2


def _build_env_and_teacher(task_id: str, checkpoint: Path, num_envs: int):
    """Build the RMA env in play mode and load a trained teacher actor.
    Args: task_id — registered task id; checkpoint — teacher .pt path; num_envs — parallel envs.
    Returns: tuple (env, runner, device).
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.scripts.train import TrainConfig
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = num_envs
    cfg.env.observations["actor"].enable_corruption = False
    cfg.env.events.pop("push_robot", None)

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(cfg.agent), "/tmp/rma_runner_tmp", device)
    runner.load(str(checkpoint))
    runner.alg.actor.eval()
    return env, runner, device


def collect_student_dataset(
    *,
    task_id: str,
    teacher_checkpoint: Path,
    out_path: Path,
    num_envs: int = 1024,
    num_steps: int = 400,
    history_length: int = 32,
) -> None:
    """Roll out the teacher and save (H, x) proprio-mass pairs to an .npz dataset.
    Args: task_id — mjlab task id; teacher_checkpoint — trained actor path; out_path — .npz destination; num_envs — parallel envs (bounds mass diversity via startup DR); num_steps — rollout length; history_length — K samples per H window.
    Returns: none.
    """
    from src.rma import RMAHistoryBuffer

    env, runner, device = _build_env_and_teacher(task_id, teacher_checkpoint, num_envs)
    obs, _ = env.reset()
    buf = RMAHistoryBuffer(num_envs, PROPRIO_DIM, history_length, device=device)

    H_list, x_list = [], []
    with torch.no_grad():
        for step in range(num_steps):
            actor_obs = obs["actor"]
            buf.push(actor_obs[:, PROPRIO_SLICE])
            if step >= history_length:
                H_list.append(buf.get().cpu().numpy())
                x_list.append(actor_obs[:, MASS_SLICE].cpu().numpy())
            actions = runner.alg.actor(obs)
            obs, _, dones, _ = env.step(actions)
            if dones.any():
                buf.reset(dones.nonzero(as_tuple=True)[0])

    H = np.concatenate(H_list, axis=0)
    x = np.concatenate(x_list, axis=0)
    print(f"Collected {H.shape[0]:_} samples — H {H.shape}, x {x.shape} "
          f"(mass mean={x.mean(0)}, std={x.std(0)})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, H=H.astype(np.float32), x=x.astype(np.float32))
    env.close()


def train_student(
    *,
    dataset_path: Path,
    out_path: Path,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_split: float = 0.1,
) -> dict:
    """Supervised MSE training of AdaptationModule on (H, x) with direct mass targets.
    Args: dataset_path — .npz with keys H and x; out_path — student .pt destination; epochs — training epochs; batch_size — SGD batch; lr — Adam learning rate; val_split — validation fraction.
    Returns: dict with train_mse, val_mse, val_mae_kg from the final epoch.
    """
    from src.rma import AdaptationModule

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = np.load(dataset_path)
    H = torch.from_numpy(data["H"]).float()
    x = torch.from_numpy(data["x"]).float()

    N = H.shape[0]
    n_val = int(N * val_split)
    perm = torch.randperm(N)
    H_tr, x_tr = H[perm[n_val:]], x[perm[n_val:]]
    H_val, x_val = H[perm[:n_val]].to(device), x[perm[:n_val]].to(device)

    proprio_dim = H.shape[-1]
    history_length = H.shape[1]
    student = AdaptationModule(proprio_dim=proprio_dim, z_dim=x.shape[-1],
                               history_length=history_length).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(H_tr, x_tr), batch_size=batch_size, shuffle=True)

    last = {}
    for ep in range(epochs):
        student.train()
        tot, nb = 0.0, 0
        for Hb, xb in loader:
            pred = student(Hb.to(device))
            loss = F.mse_loss(pred, xb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        student.eval()
        with torch.no_grad():
            val_pred = student(H_val)
            val_mse = F.mse_loss(val_pred, x_val).item()
            val_mae = (val_pred - x_val).abs().mean().item()
        last = {"train_mse": tot / nb, "val_mse": val_mse, "val_mae_kg": val_mae}
        print(f"  ep {ep+1:3d}/{epochs}  train {last['train_mse']:.5f}  val {val_mse:.5f}  "
              f"mae {val_mae:.4f} kg")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "student_state_dict": student.state_dict(),
        "proprio_dim": proprio_dim,
        "history_length": history_length,
        "z_dim": x.shape[-1],
    }, out_path)
    print(f"Saved student -> {out_path}")
    return last


class RMAPolicy(nn.Module):
    """Wrap a teacher actor and splice student mass predictions into the privileged obs slot."""

    def __init__(self, teacher_actor: nn.Module, student: nn.Module,
                 num_envs: int, history_length: int = 32,
                 device: str | torch.device = "cuda:0"):
        """Build the deploy wrapper with a per-env history buffer.
        Args: teacher_actor — trained actor; student — trained AdaptationModule; num_envs — parallel envs; history_length — proprio window K; device — buffer device.
        Returns: none.
        """
        super().__init__()
        from src.rma import RMAHistoryBuffer
        self.teacher = teacher_actor
        self.student = student
        self.buf = RMAHistoryBuffer(num_envs, PROPRIO_DIM, history_length, device=device)

    @torch.no_grad()
    def forward(self, obs) -> torch.Tensor:
        """Update the history, splice the student prediction, and run the teacher actor.
        Args: obs — rsl-rl TensorDict with an "actor" tensor.
        Returns: action tensor from the teacher.
        """
        actor_obs = obs["actor"]
        self.buf.push(actor_obs[:, PROPRIO_SLICE])
        mass_hat = self.student(self.buf.get())
        new_actor = actor_obs.clone()
        new_actor[:, MASS_SLICE] = mass_hat
        obs["actor"] = new_actor
        return self.teacher(obs)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the history buffer for the given envs.
        Args: env_ids — indices to reset, or None to reset all envs.
        Returns: none.
        """
        self.buf.reset(env_ids)


def evaluate_rma(
    *,
    task_id: str,
    teacher_checkpoint: Path,
    mode: str,
    student_checkpoint: Optional[Path] = None,
    num_envs: int = 256,
    num_steps: int = 400,
    history_length: int = 32,
) -> dict:
    """Headless rollout comparing teacher-true, RMA-predicted, and zero-mass modes.
    Args: task_id — mjlab task; teacher_checkpoint — teacher .pt; mode — one of teacher_true|rma|zero_mass; student_checkpoint — required when mode=='rma'; num_envs — parallel envs; num_steps — steps to roll out; history_length — proprio window K used by zero_mass buffer.
    Returns: dict with mode, num_envs, num_steps, ep_count, mean_return, mean_ep_length, mean_step_reward.
    """
    from src.rma import AdaptationModule, RMAHistoryBuffer

    if mode not in {"teacher_true", "rma", "zero_mass"}:
        raise ValueError(f"mode must be teacher_true|rma|zero_mass, got {mode!r}")
    if mode == "rma" and student_checkpoint is None:
        raise ValueError("mode='rma' requires student_checkpoint")

    env, runner, device = _build_env_and_teacher(task_id, teacher_checkpoint, num_envs)

    student = None
    if mode == "rma":
        payload = torch.load(str(student_checkpoint), map_location=device)
        student = AdaptationModule(
            proprio_dim=payload["proprio_dim"],
            z_dim=payload["z_dim"],
            history_length=payload["history_length"],
        ).to(device)
        student.load_state_dict(payload["student_state_dict"])
        student.eval()
        buf = RMAHistoryBuffer(num_envs, PROPRIO_DIM, payload["history_length"], device=device)
    elif mode == "zero_mass":
        buf = RMAHistoryBuffer(num_envs, PROPRIO_DIM, history_length, device=device)
    else:
        buf = None

    obs, _ = env.reset()
    reward_accum = torch.zeros(num_envs, device=device)
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    cur_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    all_step_rewards: list[float] = []

    with torch.no_grad():
        for step in range(num_steps):
            actor_obs = obs["actor"]
            if student is not None:
                buf.push(actor_obs[:, PROPRIO_SLICE])
                new_actor = actor_obs.clone()
                new_actor[:, MASS_SLICE] = student(buf.get())
                obs["actor"] = new_actor
            elif mode == "zero_mass":
                new_actor = actor_obs.clone()
                new_actor[:, MASS_SLICE] = 0.0
                obs["actor"] = new_actor

            actions = runner.alg.actor(obs)
            obs, reward, dones, _ = env.step(actions)
            reward_accum += reward
            cur_len += 1
            all_step_rewards.append(float(reward.mean().item()))
            if dones.any():
                idx = dones.nonzero(as_tuple=True)[0]
                for i in idx.tolist():
                    ep_returns.append(reward_accum[i].item())
                    ep_lengths.append(cur_len[i].item())
                reward_accum[idx] = 0.0
                cur_len[idx] = 0
                if buf is not None:
                    buf.reset(idx)

    env.close()
    return {
        "mode": mode,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "ep_count": len(ep_returns),
        "mean_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "mean_ep_length": float(np.mean(ep_lengths)) if ep_lengths else float("nan"),
        # Metric holds when num_steps < episode horizon.
        "mean_step_reward": float(np.mean(all_step_rewards)) if all_step_rewards else float("nan"),
    }
