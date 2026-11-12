# Table I：原表数值与新增 SeDPO

| Dataset | LLMs | Method | Venue | Accuracy (%) | Weighted-F1 (%) |
|---|---|---|---|---:|---:|
| OpenBookQA | Llama3-8B | Random | ACL 2022 | 72.54 | 72.7058 |
| OpenBookQA | Llama3-8B | Coverage | EMNLP 2023 | 74.80 | 75.0695 |
| OpenBookQA | Llama3-8B | SelfAdaptive | ACL 2023 | 71.60 | 72.4391 |
| OpenBookQA | Llama3-8B | DataCompression | ACL 2024 | 74.40 | 74.5382 |
| OpenBookQA | Llama3-8B | SE2 | ACL 2024 | 74.00 | 74.3878 |
| OpenBookQA | Llama3-8B | CoopES | Ours | 75.20 | 75.3462 |
| OpenBookQA | Llama3-8B | SeDPO† | NeurIPS 2025 | 89.60 | 89.6106 |
| OpenBookQA | Qwen2.5-7B | Random | ACL 2022 | 85.88 | 85.9349 |
| OpenBookQA | Qwen2.5-7B | Coverage | EMNLP 2023 | 85.60 | 85.6265 |
| OpenBookQA | Qwen2.5-7B | SelfAdaptive | ACL 2023 | 86.80 | 86.8582 |
| OpenBookQA | Qwen2.5-7B | DataCompression | ACL 2024 | 86.40 | 86.4058 |
| OpenBookQA | Qwen2.5-7B | SE2 | ACL 2024 | 87.40 | 87.4649 |
| OpenBookQA | Qwen2.5-7B | CoopES | Ours | 87.60 | 87.5913 |
| OpenBookQA | Qwen2.5-7B | SeDPO† | NeurIPS 2025 | 90.40 | 90.5061 |
| ARC-C | Llama3-8B | Random | ACL 2022 | 80.20 | 80.2511 |
| ARC-C | Llama3-8B | Coverage | EMNLP 2023 | 82.00 | 81.8013 |
| ARC-C | Llama3-8B | SelfAdaptive | ACL 2023 | 80.00 | 80.1478 |
| ARC-C | Llama3-8B | DataCompression | ACL 2024 | 82.00 | 82.0690 |
| ARC-C | Llama3-8B | SE2 | ACL 2024 | 82.00 | 81.9810 |
| ARC-C | Llama3-8B | CoopES | Ours | 82.00 | 82.3391 |
| ARC-C | Llama3-8B | SeDPO† | NeurIPS 2025 | 82.80 | 82.8012 |
| ARC-C | Qwen2.5-7B | Random | ACL 2022 | 86.60 | 86.6102 |
| ARC-C | Qwen2.5-7B | Coverage | EMNLP 2023 | 88.00 | 87.9605 |
| ARC-C | Qwen2.5-7B | SelfAdaptive | ACL 2023 | 87.80 | 87.7748 |
| ARC-C | Qwen2.5-7B | DataCompression | ACL 2024 | 87.20 | 87.4091 |
| ARC-C | Qwen2.5-7B | SE2 | ACL 2024 | 85.11 | 85.5301 |
| ARC-C | Qwen2.5-7B | CoopES | Ours | 88.40 | 88.4130 |
| ARC-C | Qwen2.5-7B | SeDPO† | NeurIPS 2025 | 88.40 | 88.3901 |

† SeDPO 为本次新运行结果，其他方法是用户提供原表中的报告值，均未重跑。虽然采用各 500 题、50 候选、k=4，但原实验的具体样本 ID 和完整推理配置未知，因此这是外部参考比较，不作为严格同设置或配对显著性证明。未完成的结果以 -- 标注；不跨来源加粗评定最佳。
