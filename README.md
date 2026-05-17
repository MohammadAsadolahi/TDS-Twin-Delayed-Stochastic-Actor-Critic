<div align="center">

# TDS — Twin Delayed Stochastic Actor-Critic

### A Novel Off-Policy Algorithm That Learns *How* to Explore

[![Paper](https://img.shields.io/badge/Research%20Paper-Preprint-blue?style=for-the-badge&logo=arxiv)](https://www.researchsquare.com/article/rs-3041837/v1)
[![PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo%20%7C%20Box2D-0081A5?style=for-the-badge)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-Research-green?style=for-the-badge)](#)

**Outperforms TD3, SAC, and DDPG across every MuJoCo and Box2D environment tested.**

Created by **[Mohammad Asadolahi](https://github.com/MohammadAsadolahi)** — Senior Agentic AI Engineer | Agentic AI Architectures In The Wild

<br>

<img src="https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/TDS.png" width="720">

</div>

---

## The Problem

State-of-the-art continuous control algorithms — TD3, SAC, DDPG — all share a fundamental limitation: **blind exploration**. They inject fixed, state-independent noise into agent decisions, hoping randomness alone will uncover optimal policies. This is catastrophically sample-inefficient and, in many real-world domains, simply infeasible.

## The Insight

Exploration should not be random. It should be **learned**.

TDS introduces a stochastic policy that outputs both a **mean action** $\mu$ and a **learned standard deviation** $\sigma$ per state-action pair. By unifying the policy gradient theorem with the deterministic policy gradient, TDS learns to *modulate its own exploration bounds* based on gradient feedback from the critics. The agent explores aggressively where it needs to, and exploits confidently where it shouldn't.

## Key Results

<div align="center">

### Last 100K Timesteps — Average Return

<img src="https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS%20-%20Table%203.png" width="720">

### Relative Performance vs. Baselines

<img src="https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS%20-%20Table%204.png" width="720">

### Learning Curves

<img src="https://github.com/MohammadAsadolahi/TDS-Twin-Delayed-Stochastic-Actor-Critic/blob/main/Plots/TDS-%20learning%20curves.png" width="720">

</div>

> Detailed per-environment learning curves (including zoomed views) are available in the [`Plots/`](Plots/) directory.

---

## Architecture

TDS combines three proven design principles with one novel contribution:

| Component | Design Choice | Rationale |
|---|---|---|
| **Twin Critics** | Two independent Q-networks, take the minimum | Mitigates overestimation bias |
| **Delayed Actor Updates** | Actor updated every 2 critic steps | Stabilizes training by reducing policy lag |
| **Stochastic Policy** *(novel)* | Actor outputs $(\mu, \sigma)$ per state | Enables gradient-driven, state-dependent exploration |
| **Target Networks** | Polyak-averaged target actor + critics | Smooth bootstrapping targets |

### Actor Network

```
State → Linear(obs_dim, 256) → LayerNorm → ReLU
      → Linear(256, 256)     → LayerNorm → ReLU
      → μ: Linear(256, action_dim) → Tanh × max_action
      → σ: Linear(256, action_dim) → Sigmoid → Clamp[0.1, 1.0] × max_action
```

The actor's $\sigma$ head learns exploration magnitude directly from policy gradient signals — no hand-tuned noise schedules, no entropy coefficients, no temperature hyperparameters.

### Critic Network (×2)

```
[State, Action] → Linear(obs_dim + action_dim, 256) → ReLU
               → Linear(256, 256) → ReLU
               → Linear(256, 1) → Q-value
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch gymnasium[mujoco] numpy pandas matplotlib
```

### 2. Train on Ant-v4

```bash
python Main.py
```

### 3. Track Progress

Evaluation runs every **5,000 steps** over 10 episodes. Training logs and results are saved to `./data/{env}/{algorithm}/seed{n}/`.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Actor Learning Rate ($\alpha$) | `3e-4` |
| Critic Learning Rate ($\beta$) | `3e-4` |
| Discount Factor ($\gamma$) | `0.99` |
| Soft Update Rate ($\tau$) | `0.005` |
| Batch Size | `100` |
| Replay Buffer Size | `1,000,000` |
| Hidden Layers | `256 × 256` |
| Actor Update Interval | Every `2` critic steps |
| Random Exploration Steps | `10,000` |
| Target Policy Noise | `0.2 × max_action` |
| Noise Clipping | `±0.5 × max_action` |

---

## Repository Structure

```
├── Actor.py                              # Stochastic actor network (μ + σ heads)
├── Critic.py                             # Twin Q-network architecture
├── Agent.py                              # TDS agent — action selection, learning, target updates
├── Replay_Buffer                         # Standard experience replay
├── Main.py                               # Training loop — Ant-v4 with evaluation
├── Requirements.py                       # Dependency list
├── TDS_solving_gymnasium_Ant_v4.ipynb    # Interactive notebook — full TDS training run
├── TDS.png                               # Architecture diagram
├── TDS_policy_architecture.png           # Policy architecture diagram
├── Benchmarks/
│   ├── DDPG_ANT_V3.ipynb                # DDPG baseline comparison
│   ├── SAC_ANT_V3.ipynb                 # SAC baseline comparison
│   └── TD3_ANT_V3.ipynb                 # TD3 baseline comparison
└── Plots/                                # All learning curves + benchmark tables
```

---

## Environments Tested

| Environment | Domain | Action Dim | Observation Dim |
|---|---|---|---|
| **Ant-v4** | MuJoCo | 8 | 27 |
| **Humanoid-v4** | MuJoCo | 17 | 376 |
| **Walker2d-v4** | MuJoCo | 6 | 17 |
| **Hopper-v4** | MuJoCo | 3 | 11 |
| **BipedalWalker-v3** | Box2D | 4 | 24 |
| **LunarLander-v2** | Box2D | 2 | 8 |

---

## Citation

```bibtex
@article{asadolahi2023tds,
  title={TDS: A Novel Stochastic Off-Policy Actor-Critic Algorithm 
         for Continuous Reinforcement Learning},
  author={Asadolahi, Mohammad},
  journal={Research Square (Preprint)},
  year={2023},
  url={https://www.researchsquare.com/article/rs-3041837/v1}
}
```

---

<div align="center">

**Questions? Feature requests?** [Open an Issue](../../issues) — contributions and discussions are welcome.

<sub>This is a research project. The author assumes no liability for deployment in production environments.</sub>

</div>
