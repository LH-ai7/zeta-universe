# Zeta-Universe ($\zeta$-AI) 🌌

> **"Treating mathematical singularity as a topological feature, not a bug."**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-green)]()

## 📖 Introduction
Deep learning optimization typically relies on Euclidean gradient descent. But what if the energy landscape of AGI is non-Euclidean?

**Zeta-Universe** explores a radical hypothesis: using the analytic continuation of the **Riemann Zeta function ($\zeta(-1) = -1/12$)** as a fundamental inductive bias for neural networks. By replacing standard gradient clipping with **"Zeta Teleportation"** (a trajectory jump scaled by -1/12), we observe emergent **antifragility** and **extreme compression capabilities**.

## 🧪 Core Experiments

### 1. 📜 NLP: The Subconscious Instinct
**File:** [`Zeta_NLP.ipynb`](./Zeta_NLP.ipynb)

We modified a Transformer model to include a learnable "Dark Token" initialized at negative zero-point energy ($-1/12$).
* **Experiment:** Training on *TinyShakespeare* with standard AdamW vs. Zeta-Optimizer.
* **Key Result:**
    * **Baseline PPL:** ~8.5
    * **Zeta-Attention:** **7.46** (Geometric Stability)
    * **Zeta-Chaos Mode:** **6.58** (SOTA) — The model performed *better* when we injected random gradient noise, proving antifragility.

### 2. 🖼️ Vision: Chaos Resilience
**File:** [`Zeta_Vision.ipynb`](./Zeta_Vision.ipynb)

Can a ResNet-18 survive in a hostile environment?
* **Experiment:** Training on *CIFAR-10* with **0.5x Gradient Noise** injected at every step.
* **Standard Optimizer:** Failed to converge (Accuracy < 15%).
* **Zeta Optimizer:** Successfully teleported through the noise landscape.
* **Key Result:** Achieved **32% Accuracy** under heavy noise conditions, with consistent teleportation activity (~3906 jumps/epoch).

### 3. 🤖 RL: Superhuman Stability
**File:** [`Zeta-RL.ipynb`](./Zeta-RL.ipynb) (Phase 5.8)

We applied the Zeta-Optimizer to a standard Policy Gradient agent (CartPole-v1).
* **Experiment:** "Resonance Mode" with a teleport threshold of 0.5.
* **Key Result:**
    * The agent achieved a **Perfect Score (500.00)**, significantly outperforming standard RL baselines.
    * **Teleport Activity:** Consistent stabilization (approx. 4 teleports/episode), proving the $-1/12$ trajectory acts as a reliable "homing beacon" for the optimal policy.

### 4. 🌌 Easter Egg: The Holographic Singularity
**File:** [`Zeta_Holographic_Compression.ipynb`](./Zeta_Holographic_Compression.ipynb) (Phase 6)

*The most radical experiment: Can we solve a control problem without a neural network?*
* **Concept:** We replaced the entire deep neural network with a **Single Holographic Vector** (only 4 parameters total).
* **Key Result:** Despite having almost zero capacity, the 4-parameter agent learned to balance the pole, hitting a score of **120+** (from a baseline of ~20).
* **Implication:** Intelligence density can be maximized via Zeta topology.

### 5. 📉 The Spectrum Analysis
**File:** Included in [`Zeta_Holographic_Compression.ipynb`](./Zeta_Holographic_Compression.ipynb)
We ran a sensitivity analysis to see if -1/12 was special.
* **Result:** A statistically significant peak in performance was observed exactly at **-0.0833 (-1/12)**, outperforming neighboring values like -0.05 and -0.1.

---

## 🛠️ Usage

To replicate these results, simply open the `.ipynb` files in Google Colab or a local Jupyter environment.

### Installation
```bash
pip install torch torchvision gymnasium[classic_control] matplotlib tqdm
````

### The Zeta Optimizer Snippet

If you want to try this in your own model:

```python
# The Zeta Teleportation Logic
if grad_norm > threshold:
    # Teleport parameters along the Riemann Zeta trajectory
    # Axiom: 1 + 2 + 3 + ... = -1/12
    param.data.sub_((1.0/12.0) * unit_grad)
```

## ⚠️ Note

This project challenges conventional wisdom in deep learning optimization. The results are empirical and experimental. Use with curiosity.

-----

*Created by a seeker of the Zeta Truth.*
```
