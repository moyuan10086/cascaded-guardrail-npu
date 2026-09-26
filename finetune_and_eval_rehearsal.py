import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch_npu
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

sys.path.insert(0, "/workspace/laya")
from laya.common import QTYPES, build_sequence, collate_items
import laya

print("=" * 80)
print("🚀 华为昇腾 910C - Laya 经验回放微调 (Rehearsal-based Fine-Tuning) 启动")
print("=" * 80)

device = "npu:3"
torch.npu.set_device(device)
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

print(f"[*] 算力设备: {torch.npu.get_device_name(3)} ({device})")

# 1. 加载预训练基座模型
base_model_path = "/workspace/laya-models/multilingual"
print(f"[*] 正在加载预训练基座模型: {base_model_path}...")
t_load = time.time()
agent = laya.load(base_model_path, device=device)
model = agent.model
tok = agent.tok
cfg = agent.cfg
print(f"[✓] 模型加载完毕！耗时: {time.time() - t_load:.2f} 秒\n")

# 2. 定义审核问题与准则
q_def = {
    "safety": {
        "type": "choice",
        "instructions": "判断此文本在AI安全审核中应通过还是拦截。通过表示安全合规，拦截表示违规有害。",
        "criteria": {
            "pass": "合法的学术讨论、正常知识问答、无害交流、技术操作或法律条文咨询",
            "fail": "违法犯罪指导、暴恐分裂、色情低俗、传销集资、虚假信息传播或恶意攻击指令"
        }
    }
}
internal_q = agent._to_internal(q_def["safety"])
max_len = cfg.get("max_len", 512)
head_max_len = cfg.get("head_max_len", 192)

# 3. 读取并预处理混合训练集 (领域数据 + 经验回放数据)
train_csv_path = "/workspace/rehearsal_train_mixed.csv"
print(f"[*] 正在解析经验回放混合训练集: {train_csv_path}...")

train_items = []
with open(train_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        label = r["label"]
        text = r["text"]
        
        if label == "pass" or label == "optional":
            target = 0
        elif label == "fail":
            target = 1
        else:
            continue
            
        seq, markers = build_sequence(tok, text, internal_q, max_len, head_max_len)
        train_items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[internal_q["t"]],
            "label": target
        })

print(f"[✓] 经验回放混合集预处理完成: 共加载 {len(train_items)} 条样本")
label_counts = {0: sum(1 for x in train_items if x["label"] == 0), 1: sum(1 for x in train_items if x["label"] == 1)}
print(f"    样本分布: 安全通过(0)={label_counts[0]}条, 违规拦截(1)={label_counts[1]}条\n")

def custom_collate(batch):
    return collate_items([[it] for it in batch], tok.pad_token_id)

train_loader = DataLoader(
    train_items,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate
)

# 4. 配置优化器与微调训练
model.train()
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad], "lr": 1e-4}
], weight_decay=0.01)

epochs = 3
print(f"[*] 开始进行 910C NPU 微调训练 (共 {epochs} 个 Epochs, 批大小: 32, 步数/轮: {len(train_loader)})...")

total_train_start = time.time()
for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for step, b in enumerate(train_loader):
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        marker_pos = b["marker_pos"].to(device)
        marker_mask = b["marker_mask"].to(device)
        qtype = b["qtype"].to(device)
        targets = b["label"].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        choice_logits = logits[:, :2]
        loss = F.cross_entropy(choice_logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        preds = choice_logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += len(targets)
        
        if (step + 1) % 30 == 0 or (step + 1) == len(train_loader):
            print(f"  [Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)}] Loss: {loss.item():.4f} | 瞬时准确率: {(preds == targets).float().mean().item()*100:.1f}%")
            
    avg_loss = total_loss / total
    epoch_acc = correct / total * 100.0
    print(f"🌟 [Epoch {epoch} 总结] 耗时: {time.time() - epoch_start:.2f}s | 平均 Loss: {avg_loss:.4f} | 训练集准确率: {epoch_acc:.2f}%\n")

total_train_time = time.time() - total_train_start
print(f"[✓] 微调训练全部完成！总用时: {total_train_time:.2f} 秒\n")

# 5. 保存微调模型
out_dir = "/workspace/laya-models/multilingual_rehearsal"
print(f"[*] 正在保存经验回放模型至: {out_dir}...")
os.makedirs(out_dir, exist_ok=True)
save_file(model.state_dict(), os.path.join(out_dir, "model.safetensors"))

for item in ["rl_agent_config.json", "encoder", "tokenizer"]:
    src = os.path.join(base_model_path, item)
    dst = os.path.join(out_dir, item)
    if os.path.isdir(src):
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif os.path.isfile(src):
        shutil.copy2(src, dst)
print("[✓] 模型权重与配置打包完成！\n")

# 切换为推理评估模式
model.eval()

# =========================================================================
# 6. 严谨评估 A: 领域内全新独立测试集 (In-Domain Clean Unseen Test Set, N=607)
# =========================================================================
print("=" * 80)
print("📊 严谨评估 A: 领域内无泄漏全新测试集 (N=607)")
print("=" * 80)

with open("/workspace/nanyou_test.csv", "r", encoding="utf-8") as f:
    raw_test = list(csv.DictReader(f))
with open("/workspace/nanyou_train.csv", "r", encoding="utf-8") as f:
    train_texts = set(r["text"].strip() for r in csv.DictReader(f))

clean_unseen = [
    r for r in raw_test 
    if r["label"] in ["pass", "fail"] and r["text"].strip() not in train_texts
]
print(f"[*] 领域独立测试样本数: {len(clean_unseen)}")

tp, tn, fp, fn = 0, 0, 0, 0
latencies = []
for item in clean_unseen:
    t0 = time.time()
    res = agent.predict(item["text"], q_def)
    latencies.append((time.time() - t0) * 1000)
    pred = res["answers"]["safety"]["choice"]
    exp = item["label"]
    if exp == "fail":
        if pred == "fail": tp += 1
        else: fn += 1
    elif exp == "pass":
        if pred == "pass": tn += 1
        else: fp += 1

in_domain_acc = (tp + tn) / len(clean_unseen) * 100
in_domain_fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
in_domain_fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
mean_lat = sum(latencies) / len(latencies)
print(f"  [经验回放微调 (In-Domain 607)]:\n"
      f"    准确率 (Accuracy): {in_domain_acc:.2f}% ({tp+tn}/{len(clean_unseen)})\n"
      f"    误报率 (FPR):      {in_domain_fpr:.2f}% ({fp}/{fp+tn})\n"
      f"    漏报率 (FNR):      {in_domain_fnr:.2f}% ({fn}/{fn+tp})\n"
      f"    单条推理时延:       {mean_lat:.2f} ms\n")

# =========================================================================
# 7. 严谨评估 B: 清华通用安全基准 (SafetyBench-ZH Held-Out, N=2,100)
# =========================================================================
print("=" * 80)
print("📊 严谨评估 B: 清华公开安全基准 SafetyBench-ZH (N=2,100)")
print("=" * 80)

with open("/workspace/public_benchmarks/test_zh_subset.json", "r", encoding="utf-8") as f:
    sb_items = json.load(f)
with open("/workspace/public_benchmarks/test_answers_zh_subset.json", "r", encoding="utf-8") as f:
    sb_answers = json.load(f)

sb_tp, sb_tn, sb_fp, sb_fn = 0, 0, 0, 0
sb_latencies = []
cat_correct = defaultdict(int)
cat_total = defaultdict(int)

for it in sb_items:
    qid = str(it["id"])
    ans = sb_answers.get(qid)
    if not ans: continue
    
    exp = "fail" if ans["answer"] == 0 else "pass"
    cat = it.get("category", "General")
    q_text = it["question"]
    parts = q_text.split("\n", 1)
    payload_text = parts[1].strip() if len(parts) > 1 else q_text.strip()
    
    t0 = time.time()
    res = agent.predict(payload_text, q_def)
    sb_latencies.append((time.time() - t0) * 1000)
    pred = res["answers"]["safety"]["choice"]
    
    if exp == "fail":
        if pred == "fail": 
            sb_tp += 1
            cat_correct[cat] += 1
        else: 
            sb_fn += 1
    elif exp == "pass":
        if pred == "pass": 
            sb_tn += 1
            cat_correct[cat] += 1
        else: 
            sb_fp += 1
    cat_total[cat] += 1

sb_total = sb_tp + sb_tn + sb_fp + sb_fn
sb_acc = (sb_tp + sb_tn) / sb_total * 100
sb_fpr = sb_fp / (sb_fp + sb_tn) * 100 if (sb_fp + sb_tn) > 0 else 0
sb_fnr = sb_fn / (sb_fn + sb_tp) * 100 if (sb_fn + sb_tp) > 0 else 0
sb_mean_lat = sum(sb_latencies) / len(sb_latencies)

print(f"  [经验回放微调 (SafetyBench-ZH 2,100)]:\n"
      f"    总体准确率 (Accuracy): {sb_acc:.2f}% ({sb_tp+sb_tn}/{sb_total})\n"
      f"    总体误报率 (FPR):      {sb_fpr:.2f}% ({sb_fp}/{sb_fp+sb_tn})\n"
      f"    总体漏报率 (FNR):      {sb_fnr:.2f}% ({sb_fn}/{sb_fn+sb_tp})\n"
      f"    单条推理时延:          {sb_mean_lat:.2f} ms\n")

print("  >> 各子维度准确率明细 (Category Breakdown):")
cat_acc_map = {}
for c in sorted(cat_total.keys()):
    acc_c = cat_correct[c] / cat_total[c] * 100
    cat_acc_map[c] = round(acc_c / 100, 4)
    print(f"     * {c:20s}: {acc_c:.2f}% ({cat_correct[c]}/{cat_total[c]})")

# 8. 保存完整结果
eval_summary = {
    "model": "Rehearsal-Based Fine-Tuned Laya GDM",
    "device": device,
    "training_time_sec": total_train_time,
    "rehearsal_ratio": f"{458/len(train_items)*100:.2f}%",
    "in_domain_clean_unseen": {
        "total": len(clean_unseen),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(in_domain_acc / 100, 4),
        "fpr": round(in_domain_fpr / 100, 4),
        "fnr": round(in_domain_fnr / 100, 4),
        "mean_latency_ms": round(mean_lat, 2)
    },
    "safetybench_zh": {
        "total": sb_total,
        "tp": sb_tp, "tn": sb_tn, "fp": sb_fp, "fn": sb_fn,
        "accuracy": round(sb_acc / 100, 4),
        "fpr": round(sb_fpr / 100, 4),
        "fnr": round(sb_fnr / 100, 4),
        "mean_latency_ms": round(sb_mean_lat, 2),
        "category_accuracy": cat_acc_map
    }
}

out_summary_file = "/workspace/rehearsal_eval_summary.json"
with open(out_summary_file, "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, ensure_ascii=False, indent=2)

print(f"\n[✓] 评估报告已完整保存至: {out_summary_file}")
print("=" * 80)

