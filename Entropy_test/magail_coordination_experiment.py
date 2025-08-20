# magail_coordination_experiment.py

import os, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
from torch import nn
from scipy.spatial.distance import jensenshannon


# =========================
# Reproducible seeding
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================
# Environment
# =========================
class CoordinationGame:
    """
    Single-state, one-step, two-agent coordination game:
    actions in {A=0, B=1}; reward +1 iff a0==a1, else 0.
    """
    def __init__(self):
        self.num_agents = 2
        self.num_actions = 2
        self.state = 0

    def reset(self):
        return {"agent_0": self.state, "agent_1": self.state}

    def step(self, actions):
        for k, a in actions.items():
            assert a in {0, 1}, f"Invalid action {a} for {k}. Must be 0 or 1."
        a0, a1 = actions["agent_0"], actions["agent_1"]
        reward = 1.0 if a0 == a1 else 0.0
        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": True, "agent_1": True, "__all__": True}
        next_obs = {"agent_0": self.state, "agent_1": self.state}
        return next_obs, rewards, dones, {}

    def get_joint_action_rewards(self):
        return {
            (a0, a1): (1.0, 1.0) if a0 == a1 else (0.0, 0.0)
            for a0 in range(self.num_actions)
            for a1 in range(self.num_actions)
        }

    def print_payoff_table(self):
        print("Payoff Table (Agent 1 rows, Agent 2 columns)")
        print("            Agent 2: A     Agent 2: B")
        for a0 in range(self.num_actions):
            row = [f"Agent 1: {'A' if a0 == 0 else 'B'}"]
            for a1 in range(self.num_actions):
                r0, r1 = self.get_joint_action_rewards()[(a0, a1)]
                row.append(f"({int(r0)}, {int(r1)})")
            print("   ".join(row))


# =========================
# Expert data generators (vectorized)
# =========================
def joint_to_index(a0, a1):
    return a0 * 2 + a1

def summarize_joint_counts(joint_actions):
    """
    joint_actions: array (N,2) with actions in {0,1}.
    Returns counts and frequencies over [AA, AB, BA, BB].
    """
    idx = joint_actions[:, 0] * 2 + joint_actions[:, 1]
    counts = np.bincount(idx, minlength=4)
    freqs = counts / counts.sum()
    return counts, freqs

def generate_expert_data(num_episodes=1000, seed=None):
    """
    Mixed independent 50/50 per agent -> joint ≈ uniform (0.25 each).
    Returns dict with vectorized arrays.
    """
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    a0 = rng.randint(0, 2, size=num_episodes)
    a1 = rng.randint(0, 2, size=num_episodes)
    actions = np.stack([a0, a1], axis=1)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Mixed 50/50 expert | counts [AA,AB,BA,BB]={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }

def generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.5, seed=None):
    """
    Correlated expert: only AA and BB with proportions AA_ratio and 1-AA_ratio.
    """
    assert 0.0 <= AA_ratio <= 1.0
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    num_AA = int(round(num_episodes * AA_ratio))
    num_BB = num_episodes - num_AA
    actions = np.empty((num_episodes, 2), dtype=int)
    actions[:num_AA] = (0, 0)  # AA
    actions[num_AA:] = (1, 1)  # BB
    rng.shuffle(actions)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Asymmetric bimodal expert | AA_ratio={AA_ratio:.2f} | counts={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }

def generate_noisy_bimodal_expert_data(num_episodes=1000, noise_level=0.1, seed=None):
    """
    Mostly AA/BB, with noise_level fraction of AB/BA.
    """
    assert 0.0 <= noise_level < 1.0
    assert num_episodes > 0
    rng = np.random.RandomState(seed)
    num_noise = int(round(num_episodes * noise_level))
    num_coord = num_episodes - num_noise
    num_AA = num_coord // 2
    num_BB = num_coord - num_AA
    actions = np.empty((num_episodes, 2), dtype=int)
    actions[:num_AA] = (0, 0)
    actions[num_AA:num_AA + num_BB] = (1, 1)
    noise_pairs = rng.randint(0, 2, size=(num_noise,))
    actions[num_AA + num_BB:] = np.stack([noise_pairs, 1 - noise_pairs], axis=1)
    rng.shuffle(actions)
    counts, freqs = summarize_joint_counts(actions)
    print(f"Noisy bimodal expert | noise={noise_level:.2f} | counts={counts.tolist()} | freqs={np.round(freqs,3)}")
    return {
        "state": np.zeros(num_episodes, dtype=int),
        "actions": actions,
        "joint_idx": actions[:, 0] * 2 + actions[:, 1]
    }


# =========================
# Policy & Discriminator
# =========================
class TabularPolicy(nn.Module):
    """
    Tabular policy over {A,B} with logits -> softmax.
    """
    def __init__(self, num_actions=2, init_uniform=True):
        super().__init__()
        self.num_actions = num_actions
        if init_uniform:
            self.logits = nn.Parameter(torch.zeros(num_actions))
        else:
            self.logits = nn.Parameter(0.1 * torch.randn(num_actions))

    def get_probs(self):
        return F.softmax(self.logits, dim=0)

    def sample_action(self):
        probs = self.get_probs()
        return torch.multinomial(probs, 1).item()

    def log_prob(self, action):
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        return F.log_softmax(self.logits, dim=0)[action]

    def entropy(self):
        probs = self.get_probs()
        log_probs = F.log_softmax(self.logits, dim=0)
        return -(probs * log_probs).sum()


class TabularDiscriminator(nn.Module):
    """
    Discriminator over joint actions (AA, AB, BA, BB) via 4 logits.
    """
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _to_idx(a0, a1):
        return a0.long() * 2 + a1.long()

    def logit(self, a0, a1):
        if not isinstance(a0, torch.Tensor):
            a0 = torch.tensor(a0, dtype=torch.long)
        if not isinstance(a1, torch.Tensor):
            a1 = torch.tensor(a1, dtype=torch.long)
        idx = self._to_idx(a0, a1).clamp(0, 3)
        return self.logits[idx]

    def forward(self, a0, a1):
        return torch.sigmoid(self.logit(a0, a1))


# =========================
# Rollout
# =========================
def collect_policy_trajectories(policies, num_episodes=100):
    """
    Roll out num_episodes one-step episodes with independent policies.
    Returns batched tensors.
    """
    env = CoordinationGame()
    a0, a1, logp0, logp1, rewards = [], [], [], [], []
    for _ in range(num_episodes):
        _ = env.reset()
        act0 = policies["agent_0"].sample_action()
        act1 = policies["agent_1"].sample_action()
        _, r, _, _ = env.step({"agent_0": act0, "agent_1": act1})
        a0.append(act0); a1.append(act1)
        logp0.append(policies["agent_0"].log_prob(act0))
        logp1.append(policies["agent_1"].log_prob(act1))
        rewards.append(r["agent_0"])
    a0 = torch.tensor(a0, dtype=torch.long)
    a1 = torch.tensor(a1, dtype=torch.long)
    joint_idx = a0 * 2 + a1
    logp0 = torch.stack(logp0)
    logp1 = torch.stack(logp1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    return {"a0": a0, "a1": a1, "joint_idx": joint_idx, "logp0": logp0, "logp1": logp1, "rewards": rewards}


# =========================
# MAGAIL Trainer
# =========================
class MAGAILTrainer:
    """
    MAGAIL for the 1-step coordination game.
    """
    def __init__(self, beta=0.0, lr_policy=0.01, lr_disc=0.01, policy_init_uniform=True,
                 reward_style="non_saturating"):
        """
        reward_style:
          - "non_saturating": r = log D
          - "gail":           r = -log(1 - D)
        """
        self.beta = beta
        self.lr_policy = lr_policy
        self.lr_disc = lr_disc
        self.reward_style = reward_style

        self.policies = {
            "agent_0": TabularPolicy(init_uniform=policy_init_uniform),
            "agent_1": TabularPolicy(init_uniform=policy_init_uniform),
        }
        self.discriminator = TabularDiscriminator()

        self.policy_optimizers = {
            "agent_0": torch.optim.Adam(self.policies["agent_0"].parameters(), lr=lr_policy),
            "agent_1": torch.optim.Adam(self.policies["agent_1"].parameters(), lr=lr_policy),
        }
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.history = {
            "policy_probs": {"agent_0": [], "agent_1": []},
            "disc_loss": [],
            "policy_loss": {"agent_0": [], "agent_1": []},
            "entropy": {"agent_0": [], "agent_1": []},
            "joint_action_dist": [],  # product joint from marginals [AA,AB,BA,BB]
        }

    def update_discriminator(self, expert_data, policy_data, batch_size=32):
        """
        Distinguish expert vs policy joint actions (BCEWithLogits).
        Works with vectorized dicts or list-of-dicts.
        """
        def _to_tensors(data, n):
            if isinstance(data, dict) and "joint_idx" in data:
                idx = torch.as_tensor(data["joint_idx"], dtype=torch.long)
                n = min(n, idx.shape[0])
                sel = torch.randperm(idx.shape[0])[:n]
                a0 = (idx[sel] // 2)
                a1 = (idx[sel] % 2)
                return a0, a1
            else:
                batch = np.random.choice(data, size=min(n, len(data)), replace=False)
                a0 = torch.tensor([t["joint_action"][0] for t in batch], dtype=torch.long)
                a1 = torch.tensor([t["joint_action"][1] for t in batch], dtype=torch.long)
                return a0, a1

        exp_a0, exp_a1 = _to_tensors(expert_data, batch_size)
        pol_a0, pol_a1 = _to_tensors(policy_data, batch_size)

        crit = torch.nn.BCEWithLogitsLoss(reduction="mean")
        exp_logits = self.discriminator.logit(exp_a0, exp_a1)
        pol_logits = self.discriminator.logit(pol_a0, pol_a1)
        ones = torch.ones_like(exp_logits)
        zeros = torch.zeros_like(pol_logits)
        disc_loss = crit(exp_logits, ones) + crit(pol_logits, zeros)

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return float(disc_loss.detach())

    def update_policies(self, policy_data, batch_size=32):
        """
        REINFORCE on discriminator-derived reward + per-agent entropy.
        """
        if isinstance(policy_data, dict) and all(k in policy_data for k in ["a0", "a1", "logp0", "logp1"]):
            N = policy_data["a0"].shape[0]
            n = min(batch_size, N)
            sel = torch.randperm(N)[:n]
            a0 = policy_data["a0"][sel]
            a1 = policy_data["a1"][sel]
            logp0 = policy_data["logp0"][sel]
            logp1 = policy_data["logp1"][sel]
        else:
            batch = np.random.choice(policy_data, size=min(batch_size, len(policy_data)), replace=False)
            a0 = torch.tensor([t["joint_action"][0] for t in batch], dtype=torch.long)
            a1 = torch.tensor([t["joint_action"][1] for t in batch], dtype=torch.long)
            logp0 = torch.stack([t["log_probs"]["agent_0"] for t in batch])
            logp1 = torch.stack([t["log_probs"]["agent_1"] for t in batch])

        with torch.no_grad():
            D = torch.sigmoid(self.discriminator.logit(a0, a1))
            if self.reward_style == "non_saturating":
                reward = torch.log(D + 1e-8)
            else:
                reward = -torch.log(1 - D + 1e-8)

        ent0 = self.policies["agent_0"].entropy()
        ent1 = self.policies["agent_1"].entropy()

        loss0 = -(logp0 * reward).mean()
        loss1 = -(logp1 * reward).mean()
        if self.beta > 0:
            loss0 = loss0 + (-self.beta * ent0)
            loss1 = loss1 + (-self.beta * ent1)

        self.policy_optimizers["agent_0"].zero_grad()
        loss0.backward(retain_graph=True)
        self.policy_optimizers["agent_0"].step()

        self.policy_optimizers["agent_1"].zero_grad()
        loss1.backward()
        self.policy_optimizers["agent_1"].step()

        return {"agent_0": float(loss0.detach()), "agent_1": float(loss1.detach())}

    def train(self, expert_data, num_epochs=500, batch_size=32, collect_every=10, rollout_episodes=100):
        for epoch in range(num_epochs):
            policy_data = collect_policy_trajectories(self.policies, num_episodes=rollout_episodes)
            disc_loss = self.update_discriminator(expert_data, policy_data, batch_size)
            policy_losses = self.update_policies(policy_data, batch_size)
            if epoch % collect_every == 0:
                self.record_statistics(disc_loss, policy_losses)

    def record_statistics(self, disc_loss, policy_losses):
        for agent in ["agent_0", "agent_1"]:
            probs = self.policies[agent].get_probs().detach().numpy()
            self.history["policy_probs"][agent].append(probs.copy())

        self.history["disc_loss"].append(disc_loss)
        for agent in ["agent_0", "agent_1"]:
            self.history["policy_loss"][agent].append(policy_losses[agent])

        for agent in ["agent_0", "agent_1"]:
            entropy = float(self.policies[agent].entropy().item())
            self.history["entropy"][agent].append(entropy)

        p0 = self.policies["agent_0"].get_probs().detach().numpy()
        p1 = self.policies["agent_1"].get_probs().detach().numpy()
        joint_dist = np.outer(p0, p1).reshape(-1)  # [AA,AB,BA,BB]
        self.history["joint_action_dist"].append(joint_dist)


# =========================
# Runner + metrics
# =========================
def _expert_num_episodes(expert_data):
    if isinstance(expert_data, dict) and "joint_idx" in expert_data:
        return int(expert_data["joint_idx"].shape[0])
    return len(expert_data)

def _expert_joint_from_data(expert_data):
    if isinstance(expert_data, dict) and "joint_idx" in expert_data:
        idx = np.asarray(expert_data["joint_idx"], dtype=int)
    else:
        idx = np.array([t["joint_action"][0] * 2 + t["joint_action"][1] for t in expert_data], dtype=int)
    counts = np.bincount(idx, minlength=4)
    return counts / counts.sum()

def _joint_hist_from_rollout(rollout):
    idx = (rollout["a0"] * 2 + rollout["a1"]).cpu().numpy()
    counts = np.bincount(idx, minlength=4)
    return counts / counts.sum()

def _coordination_rate(rollout):
    return float((rollout["a0"] == rollout["a1"]).float().mean().item())

def run_experiment(
    seeds = [42, 123, 456, 789, 999],
    beta_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
    num_epochs = 4000,
    expert_type = "bimodal",  # "mixed", "bimodal", "asymmetric", "noisy", "all_AA"
    policy_init_uniform = False,
    reward_style = "non_saturating",
    batch_size = 32,
    rollout_episodes = 200,
    eval_episodes = 5000,
    expert_seed = 0,
    lr_policy = 0.01,
    lr_disc = 0.01,
    collect_every = 10,
):
    # --- Generate expert data ---
    if expert_type == "mixed":
        expert_data = generate_expert_data(num_episodes=1000, seed=expert_seed)
    elif expert_type == "bimodal":
        expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.5, seed=expert_seed)
    elif expert_type == "asymmetric":
        expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=0.7, seed=expert_seed)
    elif expert_type == "noisy":
        expert_data = generate_noisy_bimodal_expert_data(num_episodes=1000, noise_level=0.1, seed=expert_seed)
    elif expert_type == "all_AA":
        expert_data = generate_asymmetric_bimodal_expert_data(num_episodes=1000, AA_ratio=1.0, seed=expert_seed)
    else:
        raise ValueError(f"Unknown expert_type: {expert_type}")

    n_exp = _expert_num_episodes(expert_data)
    expert_joint = _expert_joint_from_data(expert_data)
    print(f"Generated {n_exp} expert trajectories of type '{expert_type}'")
    print(f"Expert joint probs [AA,AB,BA,BB] = {np.round(expert_joint, 3)}")

    results = {}
    for beta in beta_values:
        print(f"\nRunning experiments with β = {beta}")
        results[beta] = {}
        for seed in seeds:
            print(f"  Seed {seed}...", end="")
            set_seed(seed)
            trainer = MAGAILTrainer(
                beta=beta,
                policy_init_uniform=policy_init_uniform,
                reward_style=reward_style,
                lr_policy=lr_policy,
                lr_disc=lr_disc,
            )
            trainer.train(
                expert_data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                collect_every=collect_every,
                rollout_episodes=rollout_episodes,
            )
            # Final policy state
            p0 = trainer.policies["agent_0"].get_probs().detach().numpy()
            p1 = trainer.policies["agent_1"].get_probs().detach().numpy()
            final_probs = {"agent_0": p0, "agent_1": p1}
            final_entropy = {
                "agent_0": float(trainer.policies["agent_0"].entropy().item()),
                "agent_1": float(trainer.policies["agent_1"].entropy().item()),
            }
            # Evaluation rollouts
            eval_roll = collect_policy_trajectories(trainer.policies, num_episodes=eval_episodes)
            learner_joint = _joint_hist_from_rollout(eval_roll)
            coord_rate = _coordination_rate(eval_roll)
            js_dist = float(jensenshannon(expert_joint, learner_joint, base=2))  # distance (not divergence)

            results[beta][seed] = {
                "final_probs": final_probs,
                "final_entropy": final_entropy,
                "coordination_rate": coord_rate,
                "learner_joint": learner_joint,
                "expert_joint": expert_joint,
                "js_distance": js_dist,
                "history": trainer.history,
            }
            print(" Done!")
    return results, expert_data, collect_every


# =========================
# Analysis (joint-based)
# =========================
def analyze_results(results, expert_data, use_sample_var=True, report_js_divergence=False):
    """
    Computes across-seed stats:
      - mean/variance of P(A) per agent
      - JS metric between learner JOINT and expert JOINT (distance by default)
    """
    analysis = {}
    ddof = 1 if use_sample_var else 0
    # expert joint (robust to format)
    expert_joint = _expert_joint_from_data(expert_data)

    for beta in results.keys():
        probA0, probA1, js_list = [], [], []
        for seed, res in results[beta].items():
            pA0 = float(res["final_probs"]["agent_0"][0])
            pA1 = float(res["final_probs"]["agent_1"][0])
            probA0.append(pA0); probA1.append(pA1)
            # Prefer empirical joint from evaluation
            learner_joint = np.asarray(res["learner_joint"], dtype=float)
            jsd = jensenshannon(expert_joint, learner_joint, base=2.0)  # distance
            if report_js_divergence:
                jsd = jsd**2
            js_list.append(float(jsd))

        analysis[beta] = {
            "prob_A_mean": {
                "agent_0": float(np.mean(probA0)),
                "agent_1": float(np.mean(probA1)),
            },
            "prob_A_variance": {
                "agent_0": float(np.var(probA0, ddof=ddof)),
                "agent_1": float(np.var(probA1, ddof=ddof)),
            },
            ("js_divergence" if report_js_divergence else "js_distance"): {
                "mean": float(np.mean(js_list)),
                "variance": float(np.var(js_list, ddof=ddof)),
            },
            "final_probs_all_seeds": {"agent_0": probA0, "agent_1": probA1},
            "expert_joint": expert_joint,
        }
    return analysis


# =========================
# Plotting (expert-aware)
# =========================
def plot_results(results, analysis, collect_every=10):
    """
    results: dict[beta] -> dict[seed] -> {...}
    analysis: dict containing per-beta summaries, e.g.
        analysis[beta]["final_probs_all_seeds"]["agent_0"] -> List[float]
        analysis[beta]["final_probs_all_seeds"]["agent_1"] -> List[float]
        (optionally) analysis["extremes"] -> indices or beta values to highlight
    collect_every: stride used when sampling training history (epochs)
    """
    # -------- helpers --------
    def sort_betas(betas):
        return sorted(betas, key=lambda b: float(b))

    def normalize_extremes(analysis_obj, betas_sorted):
        """
        Returns a list of beta values to highlight. Accepts:
          - analysis["extremes"] as indices (ints) or as beta values.
          - If missing/invalid, defaults to [min_beta, max_beta].
        """
        if isinstance(analysis_obj, dict) and "extremes" in analysis_obj:
            raw = analysis_obj["extremes"]
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
            norm = []
            for e in raw:
                # index path
                if isinstance(e, (int, np.integer)):
                    if 0 <= int(e) < len(betas_sorted):
                        norm.append(betas_sorted[int(e)])
                else:
                    # treat as value; map to nearest existing beta
                    try:
                        e_float = float(e)
                        idx = int(np.argmin(np.abs(np.array([float(b) for b in betas_sorted]) - e_float)))
                        norm.append(betas_sorted[idx])
                    except Exception:
                        pass
            if norm:
                return list(dict.fromkeys(norm))  # de-dup, preserve order
        # fallback: endpoints
        return betas_sorted[:1] if len(betas_sorted) == 1 else [betas_sorted[0], betas_sorted[-1]]

    def best_independent_js(expert_joint, grid=301):
        """
        Brute-force (p0, p1) on a grid in [0,1] to approximate
        the minimal JS distance between expert_joint and any product policy p0 x p1.
        """
        ps = np.linspace(0.0, 1.0, grid)
        best = np.inf
        for p0 in ps:
            for p1 in ps:
                prod = np.array(
                    [p0*p1, p0*(1-p1), (1-p0)*p1, (1-p0)*(1-p1)],
                    dtype=float
                )
                d = float(jensenshannon(expert_joint, prod, base=2.0))
                if d < best:
                    best = d
        return best


    betas_sorted = sort_betas(list(results.keys()))
    beta_values = betas_sorted  
    seeds = sorted(list(results[beta_values[0]].keys()))
    extremes = normalize_extremes(analysis, betas_sorted)

    # colormaps
    beta_cmap = mpl.colormaps.get_cmap('tab10').resampled(max(len(beta_values), 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MAGAIL Entropy Experiment Results', fontsize=16)

    # =========================
    # Plot 1: P(A) vs β (violins)
    # =========================
    ax1 = axes[0, 0]
    data0 = [analysis[b]["final_probs_all_seeds"]["agent_0"] for b in beta_values]
    data1 = [analysis[b]["final_probs_all_seeds"]["agent_1"] for b in beta_values]

    x = np.arange(len(beta_values)).astype(float)
    offset = 0.18
    width = 0.30

    v0 = ax1.violinplot(data0, positions=x - offset, widths=width,
                        showmeans=True, showextrema=False, showmedians=False)
    v1 = ax1.violinplot(data1, positions=x + offset, widths=width,
                        showmeans=True, showextrema=False, showmedians=False)

    # Color/alpha bodies to distinguish agents
    for pc in v0['bodies']:
        pc.set_alpha(0.4)
    for pc in v1['bodies']:
        pc.set_alpha(0.4)

    # Thicken mean lines (LineCollection can differ by mpl version)
    for coll in [v0.get('cmeans'), v1.get('cmeans')]:
        if coll is not None:
            try:
                coll.set_linewidths(2.0)
            except Exception:
                try:
                    coll.set_linewidth(2.0)
                except Exception:
                    pass

    ax1.axhline(0.5, linestyle='--', alpha=0.7, color='red', label='Max entropy policy')  # mixed policy
    ax1.set_xlim(-0.6, len(beta_values) - 0.4)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{b}' for b in beta_values])
    ax1.set_xlabel('β')
    ax1.set_ylabel('P(A) across seeds')
    ax1.set_title('Learning Stability')

    handles, labels = ax1.get_legend_handles_labels()
    vio1_patch = mpl.patches.Patch(alpha=0.4, label="Agent 0")
    vio2_patch = mpl.patches.Patch(color="orange", alpha=0.4, label="Agent 1")
    handles.append(vio1_patch)
    labels.append("Agent 0")
    handles.append(vio2_patch)
    labels.append("Agent 1")
    ax1.legend(handles, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # =========================
    # Plot 2: JS (joint) vs β
    # =========================
    ax2 = axes[0, 1]
    js_per_beta = []
    for b in beta_values:
        js_vals = []
        for s in seeds:
            js_vals.append(float(jensenshannon(
                np.asarray(results[b][s]["expert_joint"], dtype=float),
                np.asarray(results[b][s]["learner_joint"], dtype=float),
                base=2.0
            )))
        js_per_beta.append(js_vals)

    x = np.arange(len(beta_values)).astype(float)
    v = ax2.violinplot(js_per_beta, positions=x, widths=0.6,
                       showmeans=True, showextrema=False, showmedians=False)
    
    for pc in v['bodies']:
        pc.set_alpha(0.4)
        pc.set_facecolor("green")
    if v.get('cmeans') is not None:
        try:
            v['cmeans'].set_linewidths(2.0)
            v['cmeans'].set_color("green")
        except Exception:
            try:
                v['cmeans'].set_linewidth(2.0)
                v['cmeans'].set_color("green")
            except Exception:
                pass

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)
    indep_limit = best_independent_js(expert_joint, grid=301)
    ax2.axhline(indep_limit, color="red", linestyle='--', alpha=0.7, label='Independent-policy limit')

    ax2.set_xlabel('β')
    ax2.set_ylabel('JS distance (joint)')
    ax2.set_title('JS Distance vs β')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{b}' for b in beta_values])
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    vio_patch = mpl.patches.Patch(color="green", alpha=0.4, label="Across-seed distribution")
    handles.append(vio_patch)
    labels.append("Across-seed distribution")
    ax2.legend(handles, labels, loc='upper right')

    # =========================
    # Plot 3: Final P(A) per seed (scatter)
    # =========================
    ax3 = axes[0, 2]
    markers = ['o', 's']
    for i, b in enumerate(beta_values):
        pa0 = analysis[b]["final_probs_all_seeds"]["agent_0"]
        pa1 = analysis[b]["final_probs_all_seeds"]["agent_1"]
        jitter = 0.02
        xs = [i + np.random.uniform(-jitter, jitter) for _ in range(len(seeds))]
        ax3.scatter(xs, pa0, alpha=0.85, color=beta_cmap(i), marker=markers[0], s=50,
                    label=f'β={b} A0' if i < 2 else None)
        ax3.scatter(xs, pa1, alpha=0.85, color=beta_cmap(i), marker=markers[1], s=50,
                    label=f'β={b} A1' if i < 2 else None)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 ref')
    ax3.set_xlabel('β index (jittered)')
    ax3.set_ylabel('P(A)')
    ax3.set_title('Final Policy Probabilities Across Seeds')
    ax3.set_xticks(np.arange(len(beta_values)))
    ax3.set_xticklabels([f'{b}' for b in beta_values])
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)

    # =========================
    # Plot 4: Equilibrium selection vs β
    # =========================
    ax4 = axes[1, 0]
    eps = 0.05  # tolerance for symmetric vs collapsed

    prop_AA, prop_BB, prop_sym = [], [], []
    for b in beta_values:
        aa = bb = sym = 0
        for s in seeds:
            p0 = float(results[b][s]["final_probs"]["agent_0"][0])
            p1 = float(results[b][s]["final_probs"]["agent_1"][0])
            if p0 > 0.5 + eps and p1 > 0.5 + eps:
                aa += 1
            elif p0 < 0.5 - eps and p1 < 0.5 - eps:
                bb += 1
            else:
                sym += 1
        total = aa + bb + sym if (aa + bb + sym) > 0 else 1
        prop_AA.append(aa / total)
        prop_BB.append(bb / total)
        prop_sym.append(sym / total)

    x = np.arange(len(beta_values))
    b1 = ax4.bar(x, prop_AA, alpha=0.85, label='Collapse to AA')
    b2 = ax4.bar(x, prop_BB, bottom=prop_AA, alpha=0.85, label='Collapse to BB')
    bottom = (np.array(prop_AA) + np.array(prop_BB)).tolist()
    b3 = ax4.bar(x, prop_sym, bottom=bottom, alpha=0.85, label='Symmetric (no collapse)')

    ax4.set_xlabel('β')
    ax4.set_ylabel('Proportion of seeds')
    ax4.set_title('Equilibrium Selection Across Seeds')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{b}' for b in beta_values])
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    # =========================
    # Plot 5: Final joint vs Expert joint (extreme β)
    # =========================
    ax5 = axes[1, 1]
    action_names = ['(A,A)', '(A,B)', '(B,A)', '(B,B)']
    x = np.arange(len(action_names))
    width = 0.35

    exemplar = results[beta_values[0]][seeds[0]]
    expert_joint = np.asarray(exemplar["expert_joint"], dtype=float)

    for j, b in enumerate(extremes):
        final_joint = [np.asarray(results[b][s]["history"]["joint_action_dist"][-1], dtype=float) for s in seeds]
        mean_dist = np.mean(final_joint, axis=0)
        std_dist = np.std(final_joint, axis=0)
        offset = -width/2 if j == 0 else width/2
        ax5.bar(x + offset, mean_dist, width, alpha=0.8, yerr=std_dist, capsize=5, label=f'β={b}')
    ax5.plot(x, expert_joint, 'r--o', linewidth=2, markersize=5, label='Expert joint')
    ax5.set_xlabel('Joint Actions')
    ax5.set_ylabel('Probability')
    ax5.set_title('Final Joint Action Distribution')
    ax5.set_xticks(x)
    ax5.set_xticklabels(action_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # =========================
    # Plot 6: Policy P(A) evolution (representative seed) for extreme β
    # =========================
    ax6 = axes[1, 2]
    for b in extremes:
        seed = seeds[0]
        prob_hist = results[b][seed]["history"]["policy_probs"]["agent_0"]
        epochs = np.arange(len(prob_hist)) * collect_every
        ax6.plot(epochs, [p[0] for p in prob_hist], label=f'β={b}, P(A)', linewidth=2)
    ax6.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='0.5 ref')
    ax6.set_xlabel('Training Epoch')
    ax6.set_ylabel('P(A) for Agent 0')
    ax6.set_title('Policy Evolution During Training')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# =========================
# quick coordination summary
# =========================
def coordination_consistency(results):
    for beta in results.keys():
        coord_biases = []
        for seed in results[beta].keys():
            final_dist = np.asarray(results[beta][seed]["history"]["joint_action_dist"][-1], dtype=float)
            coord_prob = float(final_dist[0] + final_dist[3])  # AA + BB
            coord_biases.append(coord_prob)
        print(f"β={beta}: Mean coordination = {np.mean(coord_biases):.3f}, Std = {np.std(coord_biases):.3f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MAGAIL coordination experiment")
    parser.add_argument("--expert_type", type=str, default="bimodal",
                        choices=["mixed", "bimodal", "asymmetric", "noisy", "all_AA"])
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 999])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--rollout_episodes", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=5000)
    parser.add_argument("--lr_policy", type=float, default=0.01)
    parser.add_argument("--lr_disc", type=float, default=0.01)
    parser.add_argument("--reward_style", type=str, default="non_saturating",
                        choices=["non_saturating", "gail"])
    parser.add_argument("--policy_init_uniform", action="store_true",
                        help="Start policies at 0.5/0.5 instead of random")
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    print("Starting MAGAIL entropy experiment...")
    results, expert_data, collect_every = run_experiment(
        seeds=args.seeds,
        beta_values=args.betas,
        num_epochs=args.epochs,
        expert_type=args.expert_type,
        policy_init_uniform=args.policy_init_uniform,
        reward_style=args.reward_style,
        rollout_episodes=args.rollout_episodes,
        eval_episodes=args.eval_episodes,
        lr_policy=args.lr_policy,
        lr_disc=args.lr_disc,
        collect_every=10,
    )

    print("\nAnalyzing results...")
    analysis = analyze_results(results, expert_data, use_sample_var=True, report_js_divergence=False)

    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(results, analysis, collect_every=collect_every)

    print("\nCoordination consistency:")
    coordination_consistency(results)
    print("\nDone.")
