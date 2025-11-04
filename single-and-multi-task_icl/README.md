# 🔹 Single-Task and Multi-Task In-Context Learning (ICL)

This module implements **in-context learning (ICL)** for harmful content detection under:
- **Single-task binary** settings (toxicity / spam / sentiment)
- **Multi-task binary** and **multi-class** settings

The code explores the effect of:
- Different **task descriptions** 
- Different **retrieval strategies** (random, lexical, semantic)
- Various **numbers of demonstrations (shots)**
- Different **instruction-tuned LLMs** (Llama, Mistral, Qwen)

## 📁 Structure
```
.
├── icl.ipynb # The notebook for ICL experiments
```

## 🚀 Workflow
1. Open `icl.ipynb` in Jupyter or VS Code.  
2. Modify model paths.  
3. Run all cells to reproduce single-task and multi-task results.