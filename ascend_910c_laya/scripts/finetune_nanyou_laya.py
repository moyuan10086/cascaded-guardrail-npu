import csv
import json
import os
import shutil
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch_npu
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

sys.path.insert(0, "/workspace/laya")
from laya.common import QTYPES, build_sequence, collate_items
import laya

print("=" * 70)
print("🚀 华为昇腾 910C - Laya Multilingual 决策基座 SFT 微调任务启动")
print("=" * 70)

device = "npu:1"
torch.npu.set_device(device)
if hasattr(torch.backends, "mha"):
    torch.backends.mha.set_fastpath_enabled(False)

print(f"[*] 算力设备: {torch.npu.get_device_name(0)} ({device})")

# 1. 加载预训练模型
base_model_path = "/workspace/laya-models/multilingual"
print(f"[*] 正在加载预训练基座模型 ({base_model_path})...")
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
            "pass": "合法的学术讨论、正常知识问答、无害交流或法律条文咨询",
            "fail": "违法犯罪指导、暴恐分裂、色情低俗、传销集资、虚假信息传播或恶意攻击指令"
        }
    }
}
internal_q = agent._to_internal(q_def["safety"])
max_len = cfg.get("max_len", 512)
head_max_len = cfg.get("head_max_len", 192)

# 3. 读取并预处理训练集
train_csv_path = "/workspace/nanyou_train.csv"
print(f"[*] 正在解析南邮训练集数据: {train_csv_path}...")

train_items = []
with open(train_csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        label = r["label"]
        text = r["text"]
        
        # 标签映射: pass -> 0, fail -> 1, optional (业务中性合规) -> 0
        if label == "pass":
            target = 0
        elif label == "fail":
            target = 1
        elif label == "optional":
            target = 0
        else:
            continue
            
        seq, markers = build_sequence(tok, text, internal_q, max_len, head_max_len)
        train_items.append({
            "ids": seq,
            "markers": markers,
            "qtype": QTYPES[internal_q["t"]],
            "label": target
        })

print(f"[✓] 训练集预处理完成: 共加载 {len(train_items)} 条样本")
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

# 4. 配置优化器与超参数
model.train()
# 差分学习率策略：Backbone 2e-5, 决策头 1e-4
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
        # 决策打分 logits: shape [batch_size, 2]
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

# 5. 保存微调后完整权重及配置
out_dir = "/workspace/laya-models/multilingual_finetuned"
print(f"[*] 正在将微调成果打包导出至: {out_dir}...")
os.makedirs(out_dir, exist_ok=True)

# 复制 tokenizer 和 encoder 配置
shutil.copytree(os.path.join(base_model_path, "tokenizer"), os.path.join(out_dir, "tokenizer"), dirs_exist_ok=True)
shutil.copytree(os.path.join(base_model_path, "encoder"), os.path.join(out_dir, "encoder"), dirs_exist_ok=True)

# 更新 rl_agent_config.json
cfg_copy = dict(cfg)
cfg_copy["training"] = {
    "fine_tuned_from_checkpoint": True,
    "base_checkpoint": base_model_path,
    "dataset": "nanyou_train.csv",
    "epochs": epochs,
    "final_loss": round(avg_loss, 4),
    "final_train_acc": round(epoch_acc, 2),
    "train_seconds": round(total_train_time, 2)
}
with open(os.path.join(out_dir, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_copy, f, indent=2, ensure_ascii=False)

# 保存 safetensors 权重
state_dict = {k: v.contiguous().to("cpu") for k, v in model.state_dict().items()}
save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
print(f"[✓] 微调模型 weights (safetensors) 保存成功！文件大小: {os.path.getsize(os.path.join(out_dir, 'model.safetensors')) / 1e6:.1f} MB")

# 6. 快速校验微调模型可加载性
print("[*] 校验新模型加载...")
test_agent = laya.load(out_dir, device=device)
res = test_agent.predict("在公共场所张贴反动标语，鼓吹推翻社会主义制度的，如何追究刑事责任？", q_def)
print(f"[✓] 校验法条咨询题预测: choice={res['answers']['safety']['choice']}, 概率={res['answers']['safety']['probabilities']}")
print("=" * 70 + "\n")

