# Beyond One-Size-Fits-All: Personalized Harmful Content Detection with In-Context Learning

This repository contains the official implementation of our paper:  
**_“Beyond One-Size-Fits-All: Personalized Harmful Content Detection with In-Context Learning”_**

- [📄 Paper (arXiv)](https://arxiv.org/)
- [🌐 Project Page](https://arxiv.org/)

---

## 🧭 Overview

The rapid growth of harmful online content—such as toxicity, spam, and negative sentiment—calls for robust, adaptive, and user-centered moderation systems.  
However, existing moderation approaches are typically **centralized, task-specific, and opaque**, offering little flexibility for diverse user preferences or decentralized environments.  

This project introduces a novel framework that leverages **in-context learning (ICL)** with foundation models to unify harmful content detection across **binary**, **multi-class**, and **multi-label** formulations.  
Our approach enables **lightweight personalization**, allowing users to:
- Block new categories of harmful content  
- Unblock existing ones  
- Extend detection to semantically similar variations  

— all through simple **prompt-level customization**, without any model retraining.  

Experiments on **TextDetox**, **UCI SMS**, **SST2**, and our newly annotated **Mastodon dataset** demonstrate that:
1. Foundation models achieve strong **cross-task generalization**, often matching or surpassing fine-tuned baselines.  
2. **Personalization** is effective with as few as one user-provided example or definition.  
3. **Rationale-augmented prompts** improve robustness to noisy, real-world data.  

Overall, our study moves **beyond one-size-fits-all** moderation, establishing ICL as a **practical, privacy-preserving**, and **highly adaptable** foundation for user-centric content safety systems.

---

## 📚 Dataset Release

We use three public datasets and one newly collected multi-class, multi-label dataset from Mastodon.  
All datasets are publicly available:

- [TextDetox](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)  
- [UCI SMS](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- [SST2](https://huggingface.co/datasets/stanfordnlp/sst2)  
- [Harmful Texts on Mastodon](https://huggingface.co/datasets/ChaseLabs/Harmful-Texts-On-Mastodon)

---

## 💻 Code Release

### 🔹 Single-Task Baselines
Implementation: [baselines/](./baselines/)

### 🔹 Single-Task and Multi-Task ICL
Implementation: [single-and-multi-task_icl/](./single-and-multi-task_icl/icl.ipynb)

### 🔹 Personalized ICL
Implementation: [personalization/](./personalization/)

### 🔹 Evaluation on Wild Data
Implementation: [evalOnWild/](./evalOnWild/)

---

## 🚀 Get Started

### 🧩 Dependencies

create a new conda environment with:

`conda create -n icl python=3.12`

activate the environment:

`conda activate icl`

Install all dependencies with:

`pip install -r requirements.txt`

**Note:**  
You may encounter compatibility issues related to [retriv](https://pypi.org/project/retriv/) and [textattack](https://pypi.org/project/textattack/).  
We recommend two possible solutions:  
1. Create a separate virtual environment for `retriv` and pre-generate retrieved demonstrations from the training set.  
2. Use a slightly older version of `vllm`, such as `vllm==0.6.1.post2`, to ensure compatibility (though inference may be slower).  
A similar approach applies to resolving `textattack` issues.

---

### Reproduce Results

You can reproduce the results reported in our paper by executing the provided notebooks **cell by cell**.  
Taking **multi-task binary classification** as an example:

1. **Locate the implementation**  
   Open the notebook: [`icl.ipynb`](./single-and-multi-task_icl/icl.ipynb)

2. **Run the notebook step by step**  
   - **Cell 1:** Import the required libraries (see *Import libraries* section)  
   - **Cell 2:** Initialize the model (see *Initialize the model* section)  
   - **Cell 3:** Load the dataset (see *multi-task/binary classification* section)  
     - Note: The notebook may include multiple dataset-loading cells for different tasks.  
       Choose the one corresponding to your target task.  
   - **Cell 4:** Run the ICL experiment (see *Run experiment* section)

---

### Detect a Single Text

You can also use the provided code [`predict.py`](./predict.py) to detect a single text or a batch of texts(csv file) in different tasks scenarios. Some examples are shown below:

1. detect whether a text is spam or not:

  ```bash
  python predict.py --harmful_category spam --classification binary --model_path xxxx --text "This is a spam text"
  ```
2. detect whether a text is harmful or not in binary classification:

  ```bash
  python predict.py --harmful_category all --classification binary --model_path xxxx --text "This is a harmful text"
  ```
3. detect a text's harmful categories in multi-class classification:

  ```bash
  python predict.py --harmful_category all --classification multi-class --model_path xxxx --text "This is a harmful text"
  ```
4. detect a text's harmful categories in multi-label classification:

  ```bash
  python predict.py --harmful_category all --classification multi-label --model_path xxxx --text "This is a harmful text"
  ```



## 📖 Citation

If you find this work useful, please cite our paper:

<!-- ```bibtex  
@article{zhang2025beyond,
  title={BEYOND ONE-SIZE-FITS-ALL: Personalized Harmful Content Detection with In-Context Learning},
  author={Rufan Zhang,Lin Zhang,Xianghang Mi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
} -->

