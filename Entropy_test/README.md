# MAGAIL Coordination Experiment

A tiny, reproducible experiment that shows two ways MAGAIL can “fail” in a simple 2×2 coordination game—and how an entropy bonus changes the behavior.

* **Fail 1 — Symmetry/selection failure (β = 0):** with two equally-good pure NE (AA and BB), vanilla MAGAIL arbitrarily collapses to one convention depending on randomness.
* **Fail 2 — Correlated-demo mismatch:** when expert demonstrations are **correlated** (only AA/BB), MAGAIL with **independent** agent policies can’t represent that correlation. Even with entropy, it can only produce an **independent** 50/50 mix (which necessarily places mass on AB/BA), so the joint distribution never matches the demos.

With a sufficiently large entropy bonus β, both agents converge to the **max-entropy mix** (0.5/0.5) and training stabilizes across seeds.

---

## Contents

* Script: `Entropy_test/magail_coordination_experiment.py` (works as a CLI script and importable module)
* Produces: printed metrics and plots (unless `--no_plots` is set)

---

## Experiment design (what’s inside)

* **Environment:** one-step, two-agent coordination game. Reward = 1 if actions match; 0 otherwise.
* **Policies:** tabular, independent per agent (PyTorch `nn.Module` with logits; softmax → probs).
* **Discriminator:** tabular over joint actions (AA, AB, BA, BB), trained with `BCEWithLogitsLoss`.
* **Generator reward:** choose either non-saturating `log D` or GAIL `-log(1−D)`.
* **Expert datasets (vectorized, seedable):**

  * `mixed` (independent 50/50 → ≈25% each AA/AB/BA/BB)
  * `bimodal` / `asymmetric` (correlated AA/BB with chosen ratio)
  * `noisy` (mostly AA/BB + symmetric AB/BA noise)
* **Metrics:**

  * Per-agent entropy and final action probabilities
  * **Joint** action distribution $AA, AB, BA, BB$
  * **JS distance on the joint** (expert vs learner)
  * Coordination rate $P(AA)+P(BB)$
  * Across-seed variance of $P(A)$

---

## Requirements

* Python **3.12+** (PyTorch wheels are currently smoother on 3.12 than 3.13)
* Packages: `numpy`, `matplotlib`, `torch`, `scipy`

Install (in your activated virtualenv):

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scipy
# Torch (CPU) example; swap URL if you want GPU:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Don’t want SciPy? We can replace JS distance with a tiny NumPy helper later.

---

## Quick start

From the project root:

```bash
python Entropy_test/magail_coordination_experiment.py
```

Defaults:

* `--expert_type bimodal` (50% AA, 50% BB, zero AB/BA)
* `--betas 0.0 0.1 0.5 1.0 2.0 5.0`
* `--reward_style non_saturating` (`log D`)
* `--lr_policy 0.01` `--lr_disc 0.01`
* `--epochs 4000` `--seeds 42 123 456 789 999`

You’ll see training logs, plots, and a “coordination consistency” summary.

---

## CLI options

```text
--expert_type {mixed,bimodal,asymmetric,noisy,all_AA}
--betas <floats...>                 e.g., --betas 0.0 1.0 5.0
--seeds <ints...>                   e.g., --seeds 1 2 3 4 5
--epochs <int>                      training epochs (default 4000)
--rollout_episodes <int>            per-epoch rollout size (default 200)
--eval_episodes <int>               evaluation rollout size (default 5000)
--lr_policy <float>                 default 0.01
--lr_disc <float>                   default 0.01
--reward_style {non_saturating,gail}
--policy_init_uniform               start both policies at 0.5/0.5
--no_plots                          skip plotting (prints metrics only)
```

Help:

```bash
python Entropy_test/magail_coordination_experiment.py -h
```

---

## Suggested runs

### A) Show symmetry failure (β=0 collapses to a random pure convention)

```bash
python Entropy_test/magail_coordination_experiment.py \
  --expert_type bimodal \
  --betas 0.0 \
  --reward_style gail --lr_disc 0.005 \
  --policy_init_uniform \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --epochs 2000 --no_plots
```

**Expected:** coordination ≈ **1.0**; across seeds, some runs choose AA and others BB (seed-dependent collapse).

### B) Show entropy pushes to max-entropy (independent 50/50)

```bash
python Entropy_test/magail_coordination_experiment.py \
  --expert_type bimodal \
  --betas 5.0 \
  --reward_style gail --lr_disc 0.005 \
  --policy_init_uniform \
  --seeds 1 2 3 4 5 \
  --epochs 2000 --no_plots
```

**Expected:** coordination ≈ **0.5**, entropies ≈ **log 2**, per-agent $P(A)$ ≈ **0.5** across seeds.

### C) Control: correlated demos can’t be matched by independent policies

```bash
python Entropy_test/magail_coordination_experiment.py \
  --expert_type bimodal \
  --betas 0.0 5.0 \
  --reward_style gail --lr_disc 0.005 \
  --seeds 42 123 456 789 999 \
  --epochs 2000
```

**Expected (in plots):** **JS distance (joint)** stays **high** for β=0 and β=5—learner is independent, expert is correlated.

### D) Control: mixed expert (no correlation)

```bash
python Entropy_test/magail_coordination_experiment.py \
  --expert_type mixed \
  --betas 0.0 5.0 \
  --reward_style gail --lr_disc 0.005 \
  --seeds 42 123 456 789 999 \
  --epochs 2000 --no_plots
```

**Expected:** coordination ≈ **0.5** for both β; JS distance drops (learner matches uniform joint).

---

## Interpreting outputs

* **Coordination consistency:** $P(AA)+P(BB)$ from the learned policy (product joint).

  * ≈ 1.0 → near-pure corner (AA or BB).
  * ≈ 0.5 → independent 50/50 mix.

* **Policy entropy:** goes to \~0 at β=0 (pure), and to \~**log 2 ≈ 0.693** when β is large.

* **JS distance (joint):** compares expert vs learner over $AA, AB, BA, BB$.

  * For **correlated** experts (AA/BB only), it stays high even with β=5 because independent policies must put mass on AB/BA. That’s the structural mismatch we want to highlight.

* **Across-seed variance of $P(A)$:** high variance at β=0 means different runs pick different conventions. Sometimes all seeds pick the same corner by chance—bump the seed count or use `--policy_init_uniform` to make the random tie-break more visible.

---

## Reproducibility

We seed Python `random`, NumPy, PyTorch (CPU/GPU), and `PYTHONHASHSEED`. For this tabular CPU experiment, results should be stable; minor numeric differences are still possible across platforms.

---

## Troubleshooting

* **SciPy missing:**
  `python -m pip install scipy`

* **PyTorch on Python 3.13/macOS:**
  If wheels are finicky, use a Python **3.12** venv.

* **Matplotlib warning about `get_cmap`:**
  Harmless deprecation; we can switch to `plt.colormaps.get_cmap(...)` later.

---

## Citing the behavior

This experiment illustrates that:

1. **Without entropy (β=0),** MAGAIL is unstable across seeds in symmetric games and collapses to an arbitrary pure convention.
2. **With entropy (β large),** MAGAIL converges to the **max-entropy** independent mixed strategy (0.5/0.5) but **cannot** match **correlated** expert demonstrations with independent policies—joint JS stays high.

That juxtaposition is the core point we wanted for the write-up.
