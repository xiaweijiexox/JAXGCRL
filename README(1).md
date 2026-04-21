# Goal-Conditioned RL with Enhanced Critics and Cross-Episode Relabeling

This repository contains three innovations built on the [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) framework (JAX + Brax). Each variant is implemented as a set of **drop-in replacement files** for the original `crl` or `td3` agent directories.

## Contributions

### 1. Twin-Q Stabilization for Contrastive RL
Two independent contrastive critics trained in parallel; the actor optimizes against the minimum of their Q estimates. Reduces overestimation and yields up to **2× improvement** on the Ant task (0.327 → 0.645).

### 2. Cross-Episode Nearest-Neighbor Goal Relabeling (CrossE)
For each transition in the mini-batch, finds the nearest neighbor by Euclidean distance and, with probability *p*, swaps in the neighbor's relabeled goal and future supervision. Exploits the Markov property to broaden the goal distribution. Combined with Twin-Q, achieves consistent gains across all five locomotion environments (Ant Push: 0.462 → 0.620, Ant U-Maze: 0.398 → 0.529).

### 3. Quantile-Regression Distributional Critic for TD3 (TQC)
Replaces TD3's scalar critic with a quantile critic outputting *N* return quantiles, trained with quantile Huber regression. With truncation set to zero (drop=0), enriches value modeling and yields substantial improvements (Arm Push Hard: 0.010 → 0.279, Arm Reach: 0.626 → 0.796).

## File Overview

```
this_repo/
├── README.md
├── scripts/
│   ├── run_twin_q.sh          # Twin-Q CRL
│   ├── run_cross_e.sh         # Twin-Q + CrossE CRL
│   └── run_tqc.sh             # TD3 + TQC
│
├── crl_variants/
│   ├── crl_2q.py              # CRL training loop with Twin-Q critics
│   ├── crl_Markov_crossE.py   # CRL training loop with Twin-Q + CrossE
│   └── losses_2q.py           # Twin-Q contrastive losses
│
├── td3_variants/
│   ├── td3_tqc.py             # TD3 training loop with TQC hyperparameters
│   ├── losses_tqc.py          # Quantile Huber regression losses
│   └── networks_tqc.py        # QuantileCritic network
│
└── shared/
    └── networks.py             # Encoder and Actor (shared by CRL variants)
```

## Quick Start

### Prerequisites

1. Clone and install the original [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) repo.
2. Clone this repo alongside it.

### Run Scripts

Each script copies the variant files into the JaxGCRL agent directory, runs training, then restores the originals. Usage:

```bash
# Twin-Q CRL on Ant Soccer
bash scripts/run_twin_q.sh ant_soccer

# Twin-Q + CrossE CRL on Ant U-Maze (with cross_episode_ratio=0.1)
bash scripts/run_cross_e.sh ant_u_maze 0.1

# TQC on Arm Push Hard
bash scripts/run_tqc.sh arm_push_hard
```

## Results

### Locomotion (CRL-based)

| Environment | Baseline (CRL) | + Twin-Q | + Twin-Q + CrossE |
|-------------|:--------------:|:--------:|:-----------------:|
| Ant         | 0.327          | 0.645    | 0.648             |
| Ant Soccer  | 0.185          | 0.250    | 0.321             |
| Ant U-Maze  | 0.398          | 0.445    | 0.529             |
| Ant Push    | 0.462          | 0.490    | 0.620             |
| Ant Big Maze| 0.189          | 0.212    | 0.234             |

### Manipulation (TD3-based)

| Environment      | Baseline (TD3) | + QR Loss (TQC, drop=0) |
|------------------|:--------------:|:-----------------------:|
| Arm Push Hard    | 0.010          | 0.279                   |
| Arm Binpick Easy | 0.013          | 0.133                   |
| Arm Reach        | 0.626          | 0.796                   |

## Hyperparameters

### CRL + CrossE

| Parameter | Value |
|-----------|-------|
| Cross-episode probability *p* | 0.001 (Ant, Ant Soccer) / 0.1 (U-Maze, Push, Big Maze) |
| Contrastive loss | Forward InfoNCE |
| Energy function | Negative Euclidean distance |
| Representation dim | 64 |

### TD3 + TQC

| Parameter | Value |
|-----------|-------|
| Number of critics | 5 |
| Quantiles per critic | 25 |
| Quantiles to drop | 0 |
| Huber threshold κ | 1.0 |

## References

- Eysenbach et al., "Contrastive Learning as Goal-Conditioned Reinforcement Learning," NeurIPS 2022.
- Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics," ICML 2020.
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods," ICML 2018.
- Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017.
- Bortkiewicz et al., "Accelerating Goal-Conditioned RL Algorithms and Research," ICLR 2025.

## Citation

```bibtex
@misc{xia2026gcrl,
  title={Goal-Conditioned RL with Enhanced Critics and Cross-Episode Relabeling},
  author={Weijie Xia and Matthew Gawthrop},
  year={2026}
}
```
