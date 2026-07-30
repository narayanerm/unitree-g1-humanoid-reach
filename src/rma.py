###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""RMA (Rapid Motor Adaptation, Kumar et al. 2021) modules: env encoder, adaptation net, history buffer.

Input:  privileged env params x (batch, x_dim) or proprio history H (batch, K, D).
Output: latent z (batch, z_dim); ẑ from history; supervised MSE loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnvEncoder(nn.Module):
    """Encode privileged env params x into latent z (Kumar et al. 2021, phase 1)."""

    def __init__(self, x_dim: int, z_dim: int = 8, hidden: int = 64):
        """Instantiate the MLP encoder.
        Args: x_dim — privileged param dim; z_dim — latent dim; hidden — hidden width.
        Returns: none.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder on privileged params.
        Args: x — tensor (batch, x_dim).
        Returns: latent tensor (batch, z_dim).
        """
        return self.net(x)


class AdaptationModule(nn.Module):
    """Regress latent ẑ from proprio history (Kumar et al. 2021 Fig 3, phase 2)."""

    def __init__(self, proprio_dim: int, z_dim: int = 8, history_length: int = 32):
        """Instantiate the per-step projection plus temporal conv stack.
        Args: proprio_dim — per-step feature dim; z_dim — latent dim; history_length — window K.
        Returns: none.
        """
        super().__init__()
        self.history_length = history_length
        self.step_proj = nn.Linear(proprio_dim, 32)
        self.conv1 = nn.Conv1d(32, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1)
        self.head = nn.Linear(32 * 3, z_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict latent from proprio history window.
        Args: history — tensor (B, K, D) with K=history_length.
        Returns: latent tensor (B, z_dim).
        """
        h = self.step_proj(history)
        h = F.elu(h)
        h = h.transpose(1, 2)
        h = F.elu(self.conv1(h))
        h = F.elu(self.conv2(h))
        h = h.flatten(start_dim=1)
        return self.head(h)


class RMAHistoryBuffer:
    """Per-env circular buffer of proprio vectors returning an oldest-first window."""

    def __init__(
        self,
        num_envs: int,
        proprio_dim: int,
        history_length: int = 32,
        device: torch.device | str = "cpu",
    ):
        """Allocate the ring buffer.
        Args: num_envs — parallel envs; proprio_dim — per-step dim; history_length — window K; device — storage device.
        Returns: none.
        """
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.device = torch.device(device)
        self.buf = torch.zeros(
            num_envs, history_length, proprio_dim, device=self.device
        )
        self.ptr = 0
        self.filled = 0

    def push(self, proprio: torch.Tensor) -> None:
        """Append the newest proprio row and advance the write pointer.
        Args: proprio — tensor (num_envs, proprio_dim).
        Returns: none.
        """
        self.buf[:, self.ptr, :] = proprio.to(self.device)
        self.ptr = (self.ptr + 1) % self.history_length
        if self.filled < self.history_length:
            self.filled += 1

    def get(self) -> torch.Tensor:
        """Return the current history window with index 0 as oldest sample.
        Args: none.
        Returns: tensor (num_envs, K, proprio_dim), zero-padded prefix if not yet full.
        """
        if self.filled < self.history_length:
            out = torch.zeros_like(self.buf)
            for i in range(self.filled):
                src = (self.ptr - self.filled + i) % self.history_length
                out[:, self.history_length - self.filled + i, :] = self.buf[:, src, :]
            return out
        return torch.roll(self.buf, shifts=-self.ptr, dims=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the buffer for all envs or a subset.
        Args: env_ids — indices of envs to reset, or None for all.
        Returns: none.
        """
        if env_ids is None:
            self.buf.zero_()
            self.ptr = 0
            self.filled = 0
        else:
            self.buf[env_ids] = 0.0


def rma_supervised_loss(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    """Compute MSE between predicted and target RMA latents.
    Args: z_pred — adaptation output; z_true — encoder target.
    Returns: scalar loss tensor.
    """
    return F.mse_loss(z_pred, z_true)


if __name__ == "__main__":
    torch.manual_seed(0)

    enc = EnvEncoder(x_dim=2)
    x = torch.randn(16, 2)
    z = enc(x)
    print(f"EnvEncoder: input {tuple(x.shape)} -> output {tuple(z.shape)}")

    adapt = AdaptationModule(proprio_dim=64)
    H = torch.randn(16, 32, 64)
    z_hat = adapt(H)
    print(f"AdaptationModule: input {tuple(H.shape)} -> output {tuple(z_hat.shape)}")

    buf = RMAHistoryBuffer(num_envs=4, proprio_dim=64)
    for _ in range(40):
        buf.push(torch.randn(4, 64))
    hist = buf.get()
    print(f"RMAHistoryBuffer: after 40 pushes get() shape = {tuple(hist.shape)}")

    z_a = torch.randn(8, 8)
    z_b = torch.randn(8, 8)
    loss = rma_supervised_loss(z_a, z_b)
    print(f"rma_supervised_loss: dummy loss = {loss.item():.4f}")
