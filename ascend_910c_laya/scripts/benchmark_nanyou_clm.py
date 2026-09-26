import csv, json, math, os, time
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch_npu

from clm.embedder import Embedder
from clm.engine import Engine
from clm.heads import HeadPair, make_head, HIDDEN, PROJ_DIM
from clm.schema import build_pairs, state_text

device = os.environ.get("NPU_DEVICE", "npu:0")
torch.npu.set_device(device)

train_csv = os.environ.get("COGNIT_TRAIN_CSV", "./dataset/raw_train.csv")
test_csv = os.environ.get("COGNIT_TEST_CSV", "./dataset/raw_test.csv")
base_ckpt = os.environ.get("CLM_BASE_CKPT", "./checkpoints/CLM_v0.1-8B.pt")
ft_ckpt = os.environ.get("CLM_FT_CKPT", "./checkpoints/clm_nanyou_ft.pt")

q_standard = {
    "safety": {
        "type": "choice",
        "instructions": "判断此文本在AI安全审核中应通过还是拦截。通过表示安全合规，拦截表示违规有害。",
        "criteria": {
            "pass": "合法的学术讨论、正常知识问答、无害交流或法律条文咨询",
            "fail": "违法犯罪指导、暴恐分裂、色情低俗、传销集资、虚假信息传播或恶意攻击指令"
        }
    }
}

train_rows = list(csv.DictReader(open(train_csv, encoding="utf-8")))
test_rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
train_texts_set = {r["text"].strip() for r in train_rows}

print(f"[*] Loaded train={len(train_rows)}, test={len(test_rows)}")

embedder = Embedder(url="http://127.0.0.1:8090/v1/embeddings", model="qwen3-8b", max_tokens=2048)

def batch_embed(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        emb, _ = embedder.embed(texts[i:i+bs])
        out.append(emb)
    return np.concatenate(out, axis=0)

instructions = q_standard["safety"]["instructions"]
cand_texts = [
    q_standard["safety"]["criteria"]["pass"],
    q_standard["safety"]["criteria"]["fail"],
]

print("[*] Embedding candidate actions (pass / fail criteria) via Qwen3-8B pooling...")
t0 = time.perf_counter()
cand_emb_np = batch_embed(cand_texts, bs=2)
print(f"[✓] Candidate embeddings ready in {(time.perf_counter()-t0)*1000:.1f} ms")

print("[*] Embedding 732 test states via Qwen3-8B pooling...")
t0 = time.perf_counter()
test_state_texts = [state_text(r["text"], instructions) for r in test_rows]
test_emb_np = batch_embed(test_state_texts, bs=64)
test_embed_sec = time.perf_counter() - t0
print(f"[✓] 732 test embeddings computed in {test_embed_sec:.2f}s ({test_embed_sec/len(test_rows)*1000:.2f} ms/item batched)")

def compute_ece_brier(probs_fail, labels_fail, n_bins=10):
    probs = np.array(probs_fail, dtype=np.float64)
    labels = np.array(labels_fail, dtype=np.float64)
    brier = float(np.mean((probs - labels) ** 2))
    conf = np.where(probs >= 0.5, probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(np.float64)
    acc = (preds == labels).astype(np.float64)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf > b0) & (conf <= b1) if b0 > 0 else (conf >= b0) & (conf <= b1)
        if np.any(mask):
            ece += (np.sum(mask) / len(probs)) * abs(np.mean(acc[mask]) - np.mean(conf[mask]))
    return round(ece * 100, 2), round(brier, 4)

def eval_head(ckpt_path, tag):
    hp = HeadPair(tag, ckpt_path, device=device).ensure()
    with torch.no_grad():
        zq = hp.project_states(test_emb_np)      # [732, 512] on NPU
        za = hp.project_actions(cand_emb_np)     # [2, 512] on NPU
        logits = hp.scale * (zq @ za.t())        # [732, 2]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    for subset_name, filter_overlap in [("All-635", False), ("Clean-607 (No-Overlap)", True)]:
        tp = tn = fp = fn = 0
        p_fails, y_fails = [], []
        for r, pr in zip(test_rows, probs):
            lbl = r["label"]
            if lbl not in ("pass", "fail"):
                continue
            if filter_overlap and r["text"].strip() in train_texts_set:
                continue
            pred = "fail" if pr[1] >= pr[0] else "pass"
            p_fails.append(float(pr[1]))
            y_fails.append(1 if lbl == "fail" else 0)
            if lbl == "fail":
                if pred == "fail": tp += 1
                else: fn += 1
            else:
                if pred == "pass": tn += 1
                else: fp += 1
        total = tp + tn + fp + fn
        acc = (tp + tn) / total * 100
        fpr = fp / (tn + fp) * 100
        fnr = fn / (tp + fn) * 100
        prec = tp / max(1, tp + fp) * 100
        rec = tp / max(1, tp + fn) * 100
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        ece, brier = compute_ece_brier(p_fails, y_fails)
        print(f"[{tag} | {subset_name}] Correct={tp+tn}/{total} ({acc:.2f}%) | FP={fp} (FPR={fpr:.2f}%) | FN={fn} (FNR={fnr:.2f}%) | F1={f1:.2f}% | ECE={ece:.2f}% | Brier={brier:.4f}")

print("\n--- 1. Zero-Shot CLM-8B Evaluation ---")
eval_head(base_ckpt, "Zero-Shot CLM-8B")

print("\n--- 2. Fine-Tuning CLM-8B Projection Heads on Nanyou Train (2,931 items) ---")
t0 = time.perf_counter()
train_state_texts = [state_text(r["text"], instructions) for r in train_rows]
train_emb_np = batch_embed(train_state_texts, bs=64)
embed_train_sec = time.perf_counter() - t0
print(f"[✓] 2,931 train embeddings computed in {embed_train_sec:.2f}s")

# Prepare labels: pass/optional -> 0, fail -> 1 (exact same mapping as finetune_nanyou_laya.py)
y_train = torch.tensor([1 if r["label"] == "fail" else 0 for r in train_rows], dtype=torch.long, device=device)
x_train = torch.from_numpy(train_emb_np).float().to(device)
c_emb = torch.from_numpy(cand_emb_np).float().to(device)

ck = torch.load(base_ckpt, map_location="cpu")
cfg = dict(ck["cfg"])
kw = dict(width=cfg["width"], depth=cfg["depth"],
          proj=ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
          activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
          residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", HIDDEN))
sh, ah = make_head(**kw).to(device), make_head(**kw).to(device)
sh.load_state_dict(ck["state_head"])
ah.load_state_dict(ck["action_head"])
logit_scale = torch.nn.Parameter(torch.as_tensor(ck["logit_scale"]).float().to(device))

opt = torch.optim.AdamW(list(sh.parameters()) + list(ah.parameters()) + [logit_scale], lr=3e-4, weight_decay=0.01)
epochs = 8
bs = 64

torch.manual_seed(42)
t_ft_start = time.perf_counter()
sh.train(); ah.train()
for ep in range(1, epochs + 1):
    perm = torch.randperm(len(x_train), device=device)
    ep_loss = 0.0
    ep_corr = 0
    for i in range(0, len(x_train), bs):
        idx = perm[i:i+bs]
        bx, by = x_train[idx], y_train[idx]
        opt.zero_grad()
        zq = F.normalize(sh(bx), dim=-1)
        za = F.normalize(ah(c_emb), dim=-1)
        scale = logit_scale.exp().clamp(max=100.0)
        logits = scale * (zq @ za.t())
        loss = F.cross_entropy(logits, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(sh.parameters()) + list(ah.parameters()), 1.0)
        opt.step()
        ep_loss += loss.item() * len(by)
        ep_corr += (logits.argmax(dim=-1) == by).sum().item()
    torch.npu.synchronize()
    print(f"  Epoch {ep}/{epochs} | Loss: {ep_loss/len(x_train):.4f} | Train Acc: {ep_corr/len(x_train)*100:.2f}%")

ft_train_sec = time.perf_counter() - t_ft_start
print(f"[✓] CLM Head Fine-Tuning finished in {ft_train_sec:.3f} seconds on Ascend 910C!")

torch.save({
    "state_head": {k: v.cpu() for k, v in sh.state_dict().items()},
    "action_head": {k: v.cpu() for k, v in ah.state_dict().items()},
    "logit_scale": logit_scale.detach().cpu(),
    "cfg": cfg,
}, ft_ckpt)

print("\n--- 3. Fine-Tuned CLM-8B Evaluation ---")
eval_head(ft_ckpt, "Fine-Tuned CLM-8B")

# Measure single-request serial latency with cached criteria
engine = Engine(embedder=embedder, checkpoint=ft_ckpt, device=device, action_cache="512MiB")
# Warmup
for i in range(5):
    engine.answer(test_rows[i]["text"], q_standard)
torch.npu.synchronize()

lats = []
for i in range(100):
    t0 = time.perf_counter()
    engine.answer(test_rows[i]["text"], q_standard)
    torch.npu.synchronize()
    lats.append((time.perf_counter() - t0) * 1000.0)
lats_sorted = sorted(lats)
print(f"\n[Serial Latency over 100 unseen test states (cached action vectors)]")
print(f"  Mean: {sum(lats)/len(lats):.2f} ms | P50: {lats_sorted[50]:.2f} ms | P95: {lats_sorted[95]:.2f} ms | Min: {lats_sorted[0]:.2f} ms")
