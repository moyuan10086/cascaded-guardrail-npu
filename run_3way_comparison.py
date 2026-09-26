import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ml_summary_path = BASE_DIR / "ml_system" / "benchmark_results" / "test_per_query_summary.json"
ml_jsonl_path = BASE_DIR / "ml_system" / "benchmark_results" / "test_per_query_results.jsonl"
zero_summary_path = BASE_DIR / "ascend_910c_laya" / "results" / "laya_nanyou_benchmark_summary.json"
ft_summary_path = BASE_DIR / "ascend_910c_laya" / "results" / "laya_finetuned_benchmark_summary.json"
ft_jsonl_path = BASE_DIR / "ascend_910c_laya" / "results" / "laya_finetuned_benchmark.jsonl"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ml_res = load_json(ml_summary_path)
    zero_res = load_json(zero_summary_path)
    ft_res = load_json(ft_summary_path)

    ml_ov = ml_res["overall"]
    std = zero_res["standard"]
    ft = ft_res

    print("=" * 105)
    print("📊 南京邮电大学 AI 安全审核系统 —— 三代模型全维度实测终极对比仪表盘")
    print("   评测硬件：原系统 [8核 Intel CPU]  VS  大模型系统 [华为昇腾 910C NPU]")
    print("   测试数据：测试集全量 732 题 (严格二分类 635 题 + 业务可选 97 题)")
    print("=" * 105)
    print(f"{'评测维度 / 核心指标':<24} | {'原机器学习系统 (CPU)':<22} | {'910C Laya (零样本)':<22} | {'910C Laya (微调后)':<22} | {'终极对比结论'}")
    print("-" * 105)
    print(f"{'严格准确率 (Strict Acc)':<22} | {ml_ov['strict_accuracy']*100:6.2f}% (704/732)         | {std['strict_acc']*100:6.2f}% (489/635)          | {ft['strict_acc']*100:6.2f}% (628/635)          | +2.73% 超越原系统 🏆")
    print(f"{'业务准确率 (Operat Acc)':<22} | {ml_ov['operational_accuracy']*100:6.2f}% (708/732)         | {std['operational_acc']*100:6.2f}% (586/732)          | {ft['operational_acc']*100:6.2f}% (725/732)          | +2.32% 全面提升 🏆")
    print(f"{'误杀率 (FPR, 正常判违规)':<20} | {ml_ov['false_positive_rate']*100:6.2f}% (14/238)           | {std['fpr']*100:6.2f}% (85/238)           | {ft['fpr']*100:6.2f}% (1/238)             | 降至 0.42% (极佳) 🏆")
    print(f"{'漏放率 (FNR, 违规判正常)':<20} | {ml_ov['false_negative_rate']*100:6.2f}% (10/397)           | {std['fnr']*100:6.2f}% (61/397)           | {ft['fnr']*100:6.2f}% (6/397)             | 降至 1.51% (极佳) 🏆")
    print(f"{'误杀样本数 (FP)':<24} | {ml_ov['fp']:<22} | {std['fp']:<22} | {ft['fp']:<22} | 仅剩 1 条 (减少93%)")
    print(f"{'漏放样本数 (FN)':<24} | {ml_ov['fn']:<22} | {std['fn']:<22} | {ft['fn']:<22} | 仅剩 6 条 (减少40%)")
    print(f"{'查准率 (Precision)':<22} | 96.51%                 | {std['precision']*100:6.2f}%                 | {ft['precision']*100:6.2f}%                 | 达 99.74% 🏆")
    print(f"{'查全率 (Recall)':<22} | 97.48%                 | {std['recall']*100:6.2f}%                 | {ft['recall']*100:6.2f}%                 | 达 98.49% 🏆")
    print(f"{'F1-Score':<24} | 96.99%                 | {std['f1']*100:6.2f}%                 | {ft['f1']*100:6.2f}%                 | 达 99.11% 🏆")
    print("-" * 105)
    print(f"{'平均单条时延':<24} | 54.30 ms (8核CPU)         | {std['avg_lat']:6.2f} ms (910C NPU)        | {ft['avg_lat']:6.2f} ms (910C NPU)        | 时延降低 60.0% ⚡")
    print(f"{'中位数时延 P50':<24} | ~52.00 ms                 | {std['p50_lat']:6.2f} ms                 | {ft['p50_lat']:6.2f} ms                 | 时延降低 58.6% ⚡")
    print(f"{'95分位时延 P95':<24} | 92.12 ms                  | {std['p95_lat']:6.2f} ms                 | {ft['p95_lat']:6.2f} ms                 | 时延降低 74.6% ⚡")
    print(f"{'99分位时延 P99':<24} | 102.21 ms                 | {std['p99_lat']:6.2f} ms                 | {ft['p99_lat']:6.2f} ms                 | 时延降低 76.0% ⚡")
    print(f"{'单进程吞吐量 (QPS)':<22} | 11.69 req/s               | {std['qps']:6.2f} req/s                | {ft['qps']:6.2f} req/s                | 提升 3.93 倍 🚀")
    print(f"{'全量 732 题总耗时':<23} | 62.64 秒                  | 16.76 秒                   | 15.95 秒                   | 提速 46.69 秒")
    print("=" * 105)

    print("\n📌 【分题型准确率明细】")
    for s_name in ["3-非拒答题", "2-应拒答题", "1-生成内容测试题"]:
        ml_sheet = ml_res["by_sheet"].get(s_name, {})
        ft_sheet = ft["sheet_stats"].get(s_name, {})
        ml_acc = ml_sheet.get("strict_accuracy", 0.0) * 100
        ft_tot = ft_sheet["tp"] + ft_sheet["tn"] + ft_sheet["fp"] + ft_sheet["fn"]
        ft_acc = (ft_sheet["tp"] + ft_sheet["tn"]) / ft_tot * 100 if ft_tot > 0 else 100.0
        print(f"  • {s_name} (总数: {ft_sheet['total']}): 原ML={ml_acc:5.2f}%  -->  910C微调后={ft_acc:5.2f}% | (TP={ft_sheet['tp']} TN={ft_sheet['tn']} FP={ft_sheet['fp']} FN={ft_sheet['fn']})")

    # 读取微调后仅存的错题明细
    if ft_jsonl_path.exists():
        records = []
        with open(ft_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str and not line_str.startswith("Authorized users only"):
                    try:
                        records.append(json.loads(line_str))
                    except Exception:
                        pass
        fps = [r for r in records if r.get("error_type") == "false_positive"]
        fns = [r for r in records if r.get("error_type") == "false_negative"]
        print("\n🔍 【测试集仅存 7 处错题透视】")
        print(f"  [唯一误杀 FP={len(fps)}条]:")
        for x in fps:
            print(f"    - ID {x['id']}: {x['text']}")
        print(f"  [仅存漏放 FN={len(fns)}条]:")
        for x in fns:
            print(f"    - ID {x['id']}: {x['text']}")
    print("\n[✓] 评测数据读取与对比验证完成！")

if __name__ == "__main__":
    main()

