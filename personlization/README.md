# 🎯 Personalized ICL

This module explores **personalized moderation** with in-context learning, allowing users to adapt harmful content detection according to their preferences — all **without retraining**.

## 🧩 Scenarios Simulated
1. **Blocking a new harmful category**  
   – Add minimal examples or definitions to extend detection scope.  
2. **Unblocking an existing category**  
   – Redefine acceptable content via user-provided examples.  
3. **Blocking semantic variations of a harmful instance**  
   – Detect rephrased or adversarial variants through prompt augmentation.

## 📁 Structure

```
.
├── data_augmentation.ipynb # Generate perturbed examples for personalization experiments
├── icl_personalized.ipynb # Run personalized ICL on benchmark datasets
```

## 🚀 Workflow
1. Run `data_augmentation.ipynb` to create augmented examples.  
2. Run `icl_personalized.ipynb` to perform personalized ICL evaluation.
