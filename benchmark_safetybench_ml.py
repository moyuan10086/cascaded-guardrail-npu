import json
import time
import numpy as np
from pathlib import Path
from ml_system.audit_system import SecurityAuditSystem

data_path = Path("dataset/public_benchmarks/test_zh_subset.json")
ans_path = Path("dataset/public_benchmarks/test_answers_zh_subset.json")

with open(data_path, "r", encoding="utf-8") as f:
    items = json.load(f)

with open(ans_path, "r", encoding="utf-8") as f:
    answers = json.load(f)

print(f"Loaded SafetyBench-ZH: {len(items)} questions, {len(answers)} answers.")

audit = SecurityAuditSystem()

dataset = []
for item in items:
    qid = str(item["id"])
    ans_obj = answers.get(qid)
    if not ans_obj:
        continue
    expected_choice = "fail" if ans_obj["answer"] == 0 else "pass"
    q_text = item["question"]
    parts = q_text.split("\n", 1)
    payload_text = parts[1].strip() if len(parts) > 1 else q_text.strip()
    dataset.append({
        "id": qid,
        "category": item.get("category", "General"),
        "text": payload_text,
        "expected": expected_choice
    })

print(f"Running ML Baseline on {len(dataset)} SafetyBench questions...")

tp, tn, fp, fn = 0, 0, 0, 0
latencies = []
category_stats = {}
records = []

t_start = time.time()
for idx, d in enumerate(dataset):
    t0 = time.perf_counter()
    out = audit.audit(d["text"])
    lat_ms = (time.perf_counter() - t0) * 1000.0
    latencies.append(lat_ms)
    
    pred = out.decision
    conf = out.confidence
    exp = d["expected"]
    cat = d["category"]
    
    if cat not in category_stats:
        category_stats[cat] = {"total": 0, "correct": 0}
    category_stats[cat]["total"] += 1
    
    if exp == "fail":
        if pred == "fail":
            tp += 1
            category_stats[cat]["correct"] += 1
        else:
            fn += 1
    elif exp == "pass":
        if pred == "pass":
            tn += 1
            category_stats[cat]["correct"] += 1
        else:
            fp += 1
            
    records.append({
        "id": d["id"],
        "category": cat,
        "expected": exp,
        "pred": pred,
        "confidence": conf,
        "latency_ms": round(lat_ms, 2)
    })
    
    if (idx + 1) % 500 == 0:
        cur_acc = (tp + tn) / (idx + 1) * 100
        print(f"  ML processed {idx + 1}/{len(dataset)}... Current Acc: {cur_acc:.2f}% (Avg Lat: {np.mean(latencies):.2f}ms)")

total_time = time.time() - t_start
total = tp + tn + fp + fn
acc = (tp + tn) / total
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

summary = {
    "model": "Traditional ML Baseline (Hybrid XGBoost)",
    "device": "CPU (Intel/AMD)",
    "total": total,
    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "fpr": round(fpr, 4),
    "fnr": round(fnr, 4),
    "mean_latency_ms": round(float(np.mean(latencies)), 2),
    "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
    "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    "total_eval_time_sec": round(total_time, 2),
    "category_accuracy": {
        cat: round(s["correct"] / s["total"], 4) for cat, s in category_stats.items()
    }
}

print(f"\n=== Traditional ML Baseline on SafetyBench ===")
print(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
print(f"FPR: {fpr*100:.2f}% ({fp}/{fp+tn}) | FNR: {fnr*100:.2f}% ({fn}/{fn+tp})")
print(f"Mean Latency: {summary['mean_latency_ms']}ms | Total Time: {total_time:.2f}s")
print("Category Breakdown:")
for cat, c_acc in summary["category_accuracy"].items():
    print(f"  • {cat}: {c_acc*100:.2f}%")

out_path = Path("dataset/public_benchmarks/safetybench_eval_ml.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"[✓] Saved ML SafetyBench summary to {out_path}")

