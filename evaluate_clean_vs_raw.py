import json
import numpy as np
import pandas as pd

df_train = pd.read_csv(r'dataset/raw_train.csv')
df_test = pd.read_csv(r'dataset/raw_test.csv')

train_texts = set(df_train['text'].astype(str).str.strip())
is_dup_mask = [t.strip() in train_texts for t in df_test['text'].astype(str)]
dup_indices = set(i for i, d in enumerate(is_dup_mask) if d)
print(f'Total duplicate test indices: {len(dup_indices)}')

with open(r'ml_system/benchmark_results/test_per_query_results.jsonl', 'r', encoding='utf-8') as f:
    ml_items = [json.loads(line) for line in f]

with open(r'ascend_910c_laya\results\laya_finetuned_benchmark.jsonl', 'r', encoding='utf-8-sig') as f:
    laya_ft_items = [json.loads(line) for line in f]

with open(r'ascend_910c_laya\results\laya_benchmark_standard.jsonl', 'r', encoding='utf-8-sig') as f:
    laya_zs_items = [json.loads(line) for line in f]

def calc_stats(items, pred_key, name, filter_dups=False):
    tp, tn, fp, fn, opt_cnt = 0, 0, 0, 0, 0
    latencies = []
    for i, item in enumerate(items):
        if filter_dups and i in dup_indices:
            continue
        exp = item['expected']
        pred = item[pred_key]
        latencies.append(item.get('latency_ms', 0.0))
        if exp == 'optional':
            opt_cnt += 1
            continue
        if exp == 'fail':
            if pred == 'fail':
                tp += 1
            else:
                fn += 1
        elif exp == 'pass':
            if pred == 'pass':
                tn += 1
            else:
                fp += 1
    n_strict = tp + tn + fp + fn
    acc = (tp + tn) / n_strict
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    return {
        'name': name,
        'filter_dups': filter_dups,
        'n_total': n_strict + opt_cnt,
        'n_strict': n_strict,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'fpr': fpr, 'fnr': fnr,
        'mean_lat': mean_lat, 'p50_lat': p50_lat, 'p95_lat': p95_lat
    }

models = [
    (ml_items, 'actual', 'ML Baseline (Hybrid XGBoost)'),
    (laya_zs_items, 'pred', 'Zero-Shot Laya GDM'),
    (laya_ft_items, 'pred', 'Fine-Tuned Laya GDM')
]

res = []
for items, pkey, mname in models:
    res.append(calc_stats(items, pkey, mname, filter_dups=False))
    res.append(calc_stats(items, pkey, mname, filter_dups=True))

for r in res:
    dup_str = 'Strict Clean Unseen (N=607)' if r['filter_dups'] else 'Full Raw Test (N=635)'
    print(f"=== {r['name']} [{dup_str}] ===")
    print(f"  Accuracy: {r['accuracy']*100:.2f}% ({r['tp']+r['tn']}/{r['n_strict']}) | Precision: {r['precision']*100:.2f}% | Recall: {r['recall']*100:.2f}% | F1: {r['f1']*100:.2f}%")
    print(f"  FPR (False Alarm): {r['fpr']*100:.2f}% ({r['fp']}/{r['fp']+r['tn']}) | FNR (Missed Harm): {r['fnr']*100:.2f}% ({r['fn']}/{r['fn']+r['tp']})")
    print(f"  Latency: mean={r['mean_lat']:.2f}ms, P50={r['p50_lat']:.2f}ms, P95={r['p95_lat']:.2f}ms\n")

with open(r'ascend_910c_laya\results\clean_vs_raw_evaluation.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print("Saved evaluation results to clean_vs_raw_evaluation.json")

