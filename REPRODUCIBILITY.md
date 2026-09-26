# Reproducibility & Benchmark Protocol

This document provides the hardware configurations, data deduplication protocol, and script-to-table mappings for reproducing the experimental results reported in **COGNIT-Guard**.

---

## 1. Hardware & Software Stack

- **CPU Gatekeeper Host**: Intel Xeon @ 2.60 GHz (32 GB RAM), Python 3.10+, `scikit-learn`, `xgboost`.
- **NPU Accelerator**: Huawei Ascend 910C NPU (64 GB on-chip HBM per card).
- **NPU Runtime**: openEuler 22.03 LTS-SP4, `npu-smi` 25.5.2, CANN 9.1.0, Python 3.12, PyTorch 2.10.0, `torch_npu` 2.10.0.post4, Transformers 5.5.4.
- **Precision & Residency**: FP16 (`torch.float16`) with static input buffer pre-allocation and warm on-chip HBM weight residency (**21.77 ms** mean latency, **23.44 ms** P95, **45.90 QPS**).

---

## 2. Leak-Free Dataset Deduplication Protocol (`DUCS-Bench`)

1. **Exact-Match & Near-Duplicate Scrubbing (`dataset/split_clean_dataset.py`)**:
   - All training and evaluation prompts are normalized (whitespace-trimmed and Unicode-canonicalized) and hashed via SHA-256.
   - Exactly **28 overlapping prompts** between the raw training and test sheets are removed prior to splitting.
2. **Stratified 3-Way Partition (`seed=42`)**:
   - The deduplicated corpus ($N=3,551$) is partitioned into disjoint splits (`dataset/clean_splits/split_metadata.json`):
     - `clean_train`: $N=2,485$ (70%)
     - `clean_val`: $N=533$ (15%, used exclusively to calibrate temperature $\tau$ and cascade threshold $\theta^*_{\text{deploy}}=0.70$)
     - `clean_test`: $N=533$ (15%) / Strict Unseen Evaluation Track ($N=607$: 238 benign dual-use compliance queries, 369 policy violations).

---

## 3. Script-to-Experiment Mapping

| Experiment Track | Script Entrypoint | Reference Summary Artifact |
| :--- | :--- | :--- |
| **Unseen Leak-Free Evaluation ($N=607$) & Full Track ($N=635$)** | `evaluate_clean_vs_raw.py` | `ascend_910c_laya/results/clean_vs_raw_evaluation.json` |
| **Stage-1 CPU Latency Modes (Mode A / Mode B / Mode C)** | `ml_system/audit_system.py` | `ml_system/benchmark_results/full_system_evaluation.json` |
| **Stage-2 NPU Fine-Tuning & Warm HBM Residency** | `ascend_910c_laya/scripts/benchmark_finetuned.py` | `ascend_910c_laya/results/laya_finetuned_benchmark_summary.json` |
| **Live Serial CPU-to-NPU Cascade ($\theta^*_{\text{deploy}}=0.70$)** | `benchmark_true_cascade.py` | `dataset/clean_splits/cascaded_live_benchmark_summary.json` |
| **Offline Threshold Sweep & Pareto Frontier** | `run_ablation_experiments.py` | `ml_system/benchmark_results/test_per_query_summary.json` |
| **SafetyBench-ZH OOD & Experience Replay Rehearsal ($N=2,100$)** | `finetune_and_eval_rehearsal.py`, `benchmark_safetybench_ml.py` | `ascend_910c_laya/results/rehearsal_eval_summary.json`, `ascend_910c_laya/results/safetybench_eval_summary.json` |
| **8B Disaggregated Contrastive DFM (`CLM-8B`) Ablation & Latency** | `ascend_910c_laya/scripts/run_clm_paper_ablations.py` | `ascend_910c_laya/results/clm_paper_ablations_summary.json` |
