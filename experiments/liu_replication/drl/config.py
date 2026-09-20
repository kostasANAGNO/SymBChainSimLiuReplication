"""DQN hyperparameter configuration.

All values are classified per Liu 2019 fidelity labels.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    # ── Network ────────────────────────────────────────────────────────────────
    hidden_sizes: tuple[int, ...] = (128, 64)
    # Classification: OUR_RECONSTRUCTION_OF_Q_FUNCTION_PARAMETERIZATION
    # Liu specifies "DNN" without architecture; candidate-conditioned scalar Q is ours.
    activation: str = "relu"

    # ── Training ───────────────────────────────────────────────────────────────
    gamma: float = 0.9
    # Classification: OUR_RECONSTRUCTION — Liu states mu in (0,1], no numerical value.
    learning_rate: float = 1e-3
    # Classification: OUR_RECONSTRUCTION — not specified.
    weight_decay: float = 0.0
    gradient_clip_norm: float = 5.0
    # Classification: OUR_RECONSTRUCTION — stability guard.

    # ── Replay buffer ──────────────────────────────────────────────────────────
    replay_capacity: int = 2000
    # Classification: OUR_RECONSTRUCTION — not specified.
    replay_warmup_steps: int = 64
    # Classification: OUR_RECONSTRUCTION — min transitions before training.
    batch_size: int = 32
    # Classification: OUR_RECONSTRUCTION — note: distinct from consensus batch M.

    # ── Target network ─────────────────────────────────────────────────────────
    target_sync_steps: int = 20
    # Classification: OUR_RECONSTRUCTION — paper says "every G steps" (Liu Alg.1) without giving G.

    # ── Training loop ──────────────────────────────────────────────────────────
    update_every_steps: int = 1
    # Classification: OUR_RECONSTRUCTION — train once per environment step.

    # ── Epsilon-greedy ─────────────────────────────────────────────────────────
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500
    # Classification: OUR_RECONSTRUCTION — Liu specifies epsilon-greedy (Alg.1) without schedule.

    # ── Micro training ─────────────────────────────────────────────────────────
    micro_episodes: int = 300
    micro_steps_per_episode: int = 10
    # Classification: OUR_RECONSTRUCTION — micro validation budget; not a paper claim.

    # ── Normalization constants ─────────────────────────────────────────────────
    max_block_size_mb: float = 8.0
    # Classification: PAPER_EXACT (Table I: up to 8 MB)
    max_block_interval_s: float = 10.0
    # Classification: PAPER_EXACT (Table I: up to 10 s)
    max_stake: float = 50.0
    # Classification: PAPER_EXACT (Table I: 1-50 tokens)
    max_capability_ghz: float = 30.0
    # Classification: PAPER_EXACT (Table I: 10-30 GHz)
    max_link_rate_mbps: float = 100.0
    # Classification: PAPER_EXACT (Table I: 10-100 Mbps)
    max_tx_size_bytes: float = 400.0
    # Classification: OUR_RECONSTRUCTION — normalize chi; paper max is 200B but use 400 for margin.


# Single shared config instance
DEFAULT_CONFIG = DQNConfig()
