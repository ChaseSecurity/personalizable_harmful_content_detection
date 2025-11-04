# 🌍 Evaluation on Wild Data

This module evaluates ICL robustness on **wild, real-world data** collected from Mastodon — a decentralized social media platform.

## 🧠 Objectives
- Assess **cross-domain generalization** beyond clean benchmarks.  
- Explore **binary**, **multi-class**, and **multi-label** harmful content detection.  
- Study the role of **rationale-augmented prompts** for improving robustness.

## 📁 Structure
```
.
├── generate_reason.ipynb # Generate rationales for text-label pairs using DeepSeek-v3
├── icl_wild.ipynb # Evaluate ICL on Mastodon wild dataset
```

## 🚀 Workflow
1. Generate rationales with `generate_reason.ipynb`.(You need an API key from DeepSeek-v3) 
2. Evaluate ICL performance on wild data via `icl_wild.ipynb`.  