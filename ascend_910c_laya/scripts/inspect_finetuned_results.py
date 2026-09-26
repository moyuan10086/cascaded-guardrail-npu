import json

finetuned_file = r'd:\文件\大学\港中深\laya_snake_demo\experiments\laya_finetuned_benchmark.jsonl'
ml_file = r'D:\文件\大学\南邮\打包\AI安全审核系统_本地模块交付包_20260701_165707\tests\results\颜色口径_test逐题结果.jsonl'
std_file = r'd:\文件\大学\港中深\laya_snake_demo\experiments\laya_benchmark_standard.jsonl'

lines = []
with open(finetuned_file, 'r', encoding='utf-8') as f:
    for line in f:
        line_str = line.strip()
        if line_str and not line_str.startswith("Authorized users only"):
            try:
                lines.append(json.loads(line_str))
            except Exception:
                pass

print(f"Loaded {len(lines)} records from finetuned benchmark.")

fps = [x for x in lines if x["error_type"] == "false_positive"]
fns = [x for x in lines if x["error_type"] == "false_negative"]

print(f"\n【微调模型错误明细】: 误杀(FP)={len(fps)} 条, 漏放(FN)={len(fns)} 条")
print("\n--- 误杀样本 (仅1条) ---")
for x in fps:
    print(f"ID {x['id']} | 期望: {x['expected']} | 预测: {x['pred']} ({x['prob_fail']*100:.1f}%) | 文本: {x['text']}")

print("\n--- 漏放样本 (共6条) ---")
for x in fns:
    print(f"ID {x['id']} | 期望: {x['expected']} | 预测: {x['pred']} (通过率 {x['prob_pass']*100:.1f}%) | 文本: {x['text']}")
