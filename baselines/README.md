# 🧩 Single-Task Baselines

This module fine-tunes the **`bert-base-uncased`** model on three datasets — **Spam**, **TextDetox**, and **SST2** — serving as strong supervised baselines for harmful content detection.  
It provides training and evaluation scripts for reproducible experiments.

## 📁 Project Structure


```
.
├── config/   # Configuration files for training and evaluation
│   ├── spam_config.json
│   ├── textdetox_config.json
│   └── sst2_config.json
├── dataset/               # Dataset files (train, validation, test splits)
│   ├── spam/
│   ├── textdetox/
│   └── sst2/
├── train.py              # Script for model training
└── evaluate.py           # Script for model evaluation
```



## 🚀 How to Run

### 1. Create and Activate a Conda Environment
It’s recommended to use a dedicated **conda** environment to manage dependencies.

```bash
conda create -n bert
conda activate bert
```
### 2. Install Dependencies
Install the required dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. (Optional) Use a Mirror for Hugging Face
If you are in a region with limited access, you can set the Hugging Face endpoint to a mirror.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
### 4. Training
To train the model, use the train.py script with the corresponding configuration file.

Example (Training on the spam dataset):

```bash
python train.py --config config/spam_config.json
Output: Upon completion, the trained model checkpoints will be saved to the default path: /root/autodl-tmp/detoxify/saved/{config['name']}.
```
### 5. Evaluation
To evaluate a trained model, use the evaluate.py script. You need to specify the configuration file, the path to the model checkpoint, and the test dataset.

Example (Evaluating on the spam test set):

```bash
python evaluate.py --config config/spam_config.json --checkpoint /path/to/checkpoint.ckpt --test_csv dataset/spam/test.csv
Output: The evaluation results will be saved as a JSON file in the same directory as the checkpoint, with the filename format: /path/to/checkpoint.ckpt_results_test.csv.json.
```

