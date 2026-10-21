r"""Is it learning? Loss curves and weights for predictive coding (PC) and backpropagation (BP).

Uses only the saved YOLOv8n features (logs\real_features.npz). Needs numpy and matplotlib, no YOLO.
  python scripts\learning_evidence.py --npz logs\real_features.npz --out logs

Three tests, all on CLEAN features of real photos:
  1. PC alone (the project's PC layer, 256 inputs, 64 latent units, 16,384 weights): does the prediction error fall?
  2. PC vs BP on the SAME task, SAME network, SAME start weights, SAME data order.
  3. The Azure server model (Linear 256 -> 80, one Adam step per cloud frame): simulated frames vs real frames.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="logs/real_features.npz")
ap.add_argument("--out", default="logs")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
d = np.load(a.npz, allow_pickle=False)
img, dmg, spl, X, lab = d["image"], d["damage"], d["split"], d["feature"], d["label"]
cl = dmg == "clean"
order = np.argsort(img[cl]); Xc, Lc, Sc = X[cl][order], lab[cl][order].astype(int), spl[cl][order]
Xfit, Xcal, Xtest = Xc[Sc == "fit"], Xc[Sc == "cal"], Xc[Sc == "test"]
Lfit, Lcal, Ltest = Lc[Sc == "fit"], Lc[Sc == "cal"], Lc[Sc == "test"]
BLUE, TEAL, RED, GREY, AMBER = "#1f3a5f", "#17776f", "#b3402e", "#5b6570", "#b26a00"
summary = {}

# ============================================================== 1. PC alone (same update rule as extract_real_features.py)
class PC:
    def __init__(s, d=256, k=64, K=5, eta_z=0.3, lam=0.02, seed=0):
        s.W = np.random.default_rng(seed).standard_normal((d, k)) / np.sqrt(d); s.K, s.eta_z, s.lam = K, eta_z, lam
    def infer(s, x, K):
        z = np.zeros(s.W.shape[1])
        for _ in range(K):
            e = x - s.W @ z; z = z + s.eta_z * (s.W.T @ e - s.lam * z)
        return z, x - s.W @ z
    def F(s, Z, K=None):
        return float(np.mean([0.5 * (lambda e: e @ e)(s.infer(z, K or s.K)[1]) for z in Z]))
mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-6
Zfit, Zhold = (Xfit - mu) / sd, (np.vstack([Xcal, Xtest]) - mu) / sd
pc = PC(); W0 = pc.W.copy(); rng = np.random.default_rng(1)
F0_fit, F0_hold = float(np.mean(0.5 * (Zfit ** 2).sum(1))), float(np.mean(0.5 * (Zhold ** 2).sum(1)))   # predict nothing
E = 60; Ftr, Fho, dW = [pc.F(Zfit)], [pc.F(Zhold)], [0.0]
for ep in range(E):
    for i in rng.permutation(len(Zfit)):
        z, e = pc.infer(Zfit[i], 20); pc.W += 0.005 * np.outer(e, z) / np.sqrt(len(e))
        pc.W /= np.maximum(1.0, np.linalg.norm(pc.W, axis=0, keepdims=True))
    Ftr.append(pc.F(Zfit)); Fho.append(pc.F(Zhold)); dW.append(np.linalg.norm(pc.W - W0) / np.linalg.norm(W0))
summary["pc_alone"] = {"weights": int(pc.W.size), "epochs": E, "F_nothing_predicted_train": round(F0_fit, 2), "F_nothing_predicted_heldout": round(F0_hold, 2),
    "F_train_start": round(Ftr[0], 2), "F_train_end": round(Ftr[-1], 2), "F_heldout_start": round(Fho[0], 2), "F_heldout_end": round(Fho[-1], 2),
    "weight_change_rel": round(dW[-1], 3), "heldout_error_removed_pct": round(100 * (Fho[0] - Fho[-1]) / Fho[0], 1)}

fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1, 1]})
ep = np.arange(E + 1)
ax[0].plot(ep, Ftr, color=BLUE, label="train photos (39)"); ax[0].plot(ep, Fho, color=TEAL, label="held-out photos (89)")
ax[0].axhline(F0_hold, color=GREY, ls=":", label="predict nothing"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("prediction error F (loss)"); ax[0].set_title("PC: the loss falls"); ax[0].legend(fontsize=8)
ax[1].plot(ep, dW, color=AMBER); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("‖W − W₀‖ / ‖W₀‖"); ax[1].set_title("PC: the weights move")
vm = np.abs(pc.W).max()
ax[2].imshow(W0, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm); ax[2].set_title("weights before (random)"); ax[2].set_xlabel("64 latent units"); ax[2].set_ylabel("256 inputs")
im = ax[3].imshow(pc.W, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm); ax[3].set_title("weights after training"); ax[3].set_xlabel("64 latent units")
fig.colorbar(im, ax=ax[3], fraction=0.05); fig.tight_layout(); fig.savefig(out / "learning_1_pc_alone.png", dpi=150); plt.close(fig)

# ============================================================== 2. PC vs BP, same task / network / start / order
Xtr_raw, Ltr = np.vstack([Xfit, Xcal]), np.r_[Lfit, Lcal]
m2, s2 = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
Xtr, Xte = (Xtr_raw - m2) / s2, (Xtest - m2) / s2
Ytr, Yte = np.eye(80)[Ltr], np.eye(80)[Ltest]
H = 64; r0 = np.random.default_rng(a.seed)
W1_0 = r0.standard_normal((H, 256)) / np.sqrt(256); W2_0 = r0.standard_normal((80, H)) / np.sqrt(H)
f = np.tanh; fp = lambda x: 1 - np.tanh(x) ** 2
def fwd(W1, W2, x): h1 = W1 @ x; return h1, W2 @ f(h1)
def mse_acc(W1, W2, X_, Y_):
    P = np.array([fwd(W1, W2, x)[1] for x in X_]); return float(np.mean(0.5 * ((Y_ - P) ** 2).sum(1))), float(np.mean(P.argmax(1) == Y_.argmax(1)))
def grad_bp(W1, W2, x, y):
    h1, o = fwd(W1, W2, x); e2 = y - o; e1 = (W2.T @ e2) * fp(h1)
    return np.outer(e1, x), np.outer(e2, f(h1))            # = -dL/dW for L = 0.5*||y-o||^2
def grad_pc(W1, W2, x, y, T=60, eta=0.1):
    mu1 = W1 @ x; x1 = mu1.copy()
    for _ in range(T):
        e1 = x1 - mu1; e2 = y - W2 @ f(x1); x1 = x1 + eta * (-e1 + (W2.T @ e2) * fp(x1))
    e1 = x1 - mu1; e2 = y - W2 @ f(x1)
    return np.outer(e1, x), np.outer(e2, f(x1))
lr, E2 = 0.02, 150; nets = {k: [W1_0.copy(), W2_0.copy()] for k in ("BP", "PC")}
hist = {k: {"tr": [], "te": [], "acc": [], "dw": []} for k in nets}; cos = []
def rec(k):
    W1, W2 = nets[k]; tr, _ = mse_acc(W1, W2, Xtr, Ytr); te, ac = mse_acc(W1, W2, Xte, Yte)
    hist[k]["tr"].append(tr); hist[k]["te"].append(te); hist[k]["acc"].append(ac)
    hist[k]["dw"].append(np.sqrt(np.sum((W1 - W1_0) ** 2) + np.sum((W2 - W2_0) ** 2)) / np.sqrt(np.sum(W1_0 ** 2) + np.sum(W2_0 ** 2)))
for k in nets: rec(k)
rp = np.random.default_rng(3)
for ep in range(E2):
    perm = rp.permutation(len(Xtr))
    # how closely does a PC update match the BP gradient? measured on the PC network's own weights, 20 photos
    cs = []
    for i in perm[:20]:
        g1b, g2b = grad_bp(*nets["PC"], Xtr[i], Ytr[i]); g1p, g2p = grad_pc(*nets["PC"], Xtr[i], Ytr[i])
        u, v = np.r_[g1b.ravel(), g2b.ravel()], np.r_[g1p.ravel(), g2p.ravel()]; cs.append(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    cos.append(float(np.mean(cs)))
    for i in perm:
        for k, g in (("BP", grad_bp), ("PC", grad_pc)):
            W1, W2 = nets[k]; g1, g2 = g(W1, W2, Xtr[i], Ytr[i]); W1 += lr * g1; W2 += lr * g2
    for k in nets: rec(k)
maj = float(np.mean(Ltest == np.bincount(Ltr).argmax())); mse_prior = float(np.mean(0.5 * ((Yte - Ytr.mean(0)) ** 2).sum(1)))
summary["pc_vs_bp"] = {"train_photos": len(Xtr), "test_photos": len(Xte), "epochs": E2, "lr": lr,
    "BP": {"train_loss_start": round(hist["BP"]["tr"][0], 3), "train_loss_end": round(hist["BP"]["tr"][-1], 4), "test_loss_end": round(hist["BP"]["te"][-1], 3), "test_top1": round(hist["BP"]["acc"][-1], 3)},
    "PC": {"train_loss_start": round(hist["PC"]["tr"][0], 3), "train_loss_end": round(hist["PC"]["tr"][-1], 4), "test_loss_end": round(hist["PC"]["te"][-1], 3), "test_top1": round(hist["PC"]["acc"][-1], 3)},
    "BP_best_test_loss": [round(float(min(hist["BP"]["te"])), 3), int(np.argmin(hist["BP"]["te"]))], "PC_best_test_loss": [round(float(min(hist["PC"]["te"])), 3), int(np.argmin(hist["PC"]["te"]))], "baseline_always_most_common_class_top1": round(maj, 3), "baseline_class_frequency_test_loss": round(mse_prior, 3),
    "cosine_PC_update_vs_BP_gradient": {"first_epoch": round(cos[0], 3), "last_epoch": round(cos[-1], 3), "mean": round(float(np.mean(cos)), 3)}}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4)); e2 = np.arange(E2 + 1)
for k, c, ls in (("BP", BLUE, "-"), ("PC", TEAL, "--")): ax[0].plot(e2, hist[k]["tr"], color=c, ls=ls, label=k)
ax[0].set_title("training loss (same task)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
for k, c, ls in (("BP", BLUE, "-"), ("PC", TEAL, "--")): ax[1].plot(e2, hist[k]["te"], color=c, ls=ls, label=k)
ax[1].axhline(mse_prior, color=GREY, ls=":", label="class-frequency guess"); ax[1].set_title("loss on 65 unseen photos"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
for k, c, ls in (("BP", BLUE, "-"), ("PC", TEAL, "--")): ax[2].plot(e2, hist[k]["dw"], color=c, ls=ls, label=k)
ax[2].set_title("weights move: ‖W − W₀‖ / ‖W₀‖"); ax[2].set_xlabel("epoch"); ax[2].legend()
ax[3].plot(np.arange(1, E2 + 1), cos, color=AMBER); ax[3].set_ylim(0, 1.02); ax[3].set_title("PC update vs BP gradient (cosine)"); ax[3].set_xlabel("epoch"); ax[3].axhline(1, color=GREY, ls=":")
fig.tight_layout(); fig.savefig(out / "learning_2_pc_vs_bp.png", dpi=150); plt.close(fig)

# ============================================================== 3. the Azure model: Linear(256,80), CE, one Adam step per frame
def adam_run(feats, labels, steps, lr=1e-3, seed=0):
    r = np.random.default_rng(seed); bound = 1 / np.sqrt(256)
    W = r.uniform(-bound, bound, (80, 256)); b = r.uniform(-bound, bound, 80); m = [np.zeros_like(W), np.zeros_like(b)]; v = [np.zeros_like(W), np.zeros_like(b)]
    W0_, b0_ = W.copy(), b.copy(); losses = []
    for t in range(1, steps + 1):
        i = (t - 1) % len(feats); x, y = feats[i], labels[i]
        z = W @ x + b; z -= z.max(); p = np.exp(z) / np.exp(z).sum(); losses.append(float(-np.log(p[y] + 1e-12)))
        g = p.copy(); g[y] -= 1; gW, gb = np.outer(g, x), g
        for j, (P, G) in enumerate(((W, gW), (b, gb))):
            m[j] = 0.9 * m[j] + 0.1 * G; v[j] = 0.999 * v[j] + 0.001 * G ** 2
            P -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
    return np.array(losses), W - W0_, b - b0_, (W, b), (W0_, b0_)
S = 400; rs = np.random.default_rng(5)
sim_f = rs.standard_normal((S, 256)); sim_l = np.zeros(S, int)                    # what the simulation sent: random numbers, label 0
real_f, real_l = Xtr_raw, Ltr                                                      # what real mode sends: YOLO features, YOLO label
Ls, dWs, dbs, _, _ = adam_run(sim_f, sim_l, S); Lr_, dWr, dbr, (Wr, br), (Wi, bi) = adam_run(real_f, real_l, S)
def heldout_ce(W, b, F_, L_):
    z = F_ @ W.T + b; z -= z.max(1, keepdims=True); p = np.exp(z) / np.exp(z).sum(1, keepdims=True); return float(np.mean(-np.log(p[np.arange(len(L_)), L_] + 1e-12)))
prior = np.bincount(Ltr, minlength=80) + 1.0; prior /= prior.sum()
summary["azure_model"] = {"steps": S, "lr_assumed": 1e-3, "chance_loss_ln80": round(float(np.log(80)), 3),
    "simulated_random_features_label0": {"loss_first10": round(float(Ls[:10].mean()), 3), "loss_last10": round(float(Ls[-10:].mean()), 4), "loss_after_56_steps": round(float(Ls[46:56].mean()), 3)},
    "real_features_yolo_labels": {"loss_first10": round(float(Lr_[:10].mean()), 3), "loss_last10": round(float(Lr_[-10:].mean()), 4)},
    "real_heldout_loss_before": round(heldout_ce(Wi, bi, Xtest, Ltest), 3), "real_heldout_loss_after": round(heldout_ce(Wr, br, Xtest, Ltest), 3), "real_heldout_loss_class_frequency_guess": round(float(np.mean(-np.log(prior[Ltest]))), 3)}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4)); sm = lambda x, w=10: np.convolve(x, np.ones(w) / w, mode="valid")
ax[0].plot(sm(Ls), color=RED, label="simulation: random numbers, label 0"); ax[0].plot(sm(Lr_), color=TEAL, label="real: YOLOv8n features, YOLO label"); ax[0].axhline(np.log(80), color=GREY, ls=":", label="chance (ln 80)")
ax[0].set_xlabel("cloud step (1 frame = 1 step)"); ax[0].set_ylabel("loss (10-step average)"); ax[0].set_title("Azure model: loss per step"); ax[0].legend(fontsize=8)
ax[1].bar(["before\ntraining", "after\n400 steps", "class-frequency\nguess"], [heldout_ce(Wi, bi, Xtest, Ltest), heldout_ce(Wr, br, Xtest, Ltest), float(np.mean(-np.log(prior[Ltest])))], color=[GREY, TEAL, AMBER]); ax[1].set_title("real run: loss on 65 unseen photos"); ax[1].set_ylabel("loss")
ax[2].hist(dWr.ravel(), bins=60, color=TEAL); ax[2].set_title("real run: change of the 20,480 weights"); ax[2].set_xlabel("W − W₀")
fig.tight_layout(); fig.savefig(out / "learning_3_azure_model.png", dpi=150); plt.close(fig)
json.dump(summary, open(out / "learning_evidence.json", "w"), indent=2); print(json.dumps(summary, indent=2))