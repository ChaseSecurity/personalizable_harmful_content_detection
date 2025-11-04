import argparse
import json
import os
import sys
import warnings

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import src.data_loaders as module_data
import torch
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from src.data_loaders import JigsawDataBias, JigsawDataMultilingual, JigsawDataOriginal
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import ToxicClassifier


def calculate_metrics(y_true, y_pred):
    """计算各项评估指标"""
    # 处理可能的除零警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 准确率
        acc = accuracy_score(y_true, y_pred)
        
        # 精确率
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        # 召回率
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # F1分数
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 计算假阳性率
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr
    }


def test_classifier(config, dataset, checkpoint_path, device="cuda:0"):
    model = ToxicClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    test_dataset = get_instance(module_data, "dataset", config, train=False)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=20,
        shuffle=False,
    )

    scores = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        if "multi_target" in meta:
            targets += meta["multi_target"]
        else:
            targets += meta["target"]

        ids += meta["text_id"]
        with torch.no_grad():
            out = model.forward(*items)
            # TODO: save embeddings
            sm = torch.sigmoid(out).cpu().detach().numpy()
        scores.extend(sm)

    # 转换为二值预测（阈值0.5）
    binary_scores = [s >= 0.5 for s in scores]
    binary_scores = np.stack(binary_scores)
    scores = np.stack(scores)
    targets = np.stack(targets)
    
    # 初始化所有评估指标的列表
    auc_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    fpr_scores = []

    # 为每个类别计算指标
    for class_idx in range(scores.shape[1]):
        # 过滤掉标记为-1的样本（不参与评估）
        mask = targets[:, class_idx] != -1
        target_binary = targets[mask, class_idx]
        class_scores = scores[mask, class_idx]
        class_preds = binary_scores[mask, class_idx]
        
        # 计算AUC
        try:
            auc = roc_auc_score(target_binary, class_scores)
            auc_scores.append(auc)
        except Exception:
            warnings.warn(
                "Only one class present in y_true. ROC AUC score is not defined in that case. Set to nan for now."
            )
            auc_scores.append(np.nan)
        
        # 计算其他指标
        if len(np.unique(target_binary)) < 2:
            # 如果只有一个类别，无法计算部分指标
            accuracy_scores.append(np.nan)
            precision_scores.append(np.nan)
            recall_scores.append(np.nan)
            f1_scores.append(np.nan)
            fpr_scores.append(np.nan)
        else:
            metrics = calculate_metrics(target_binary, class_preds)
            accuracy_scores.append(metrics['accuracy'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            fpr_scores.append(metrics['false_positive_rate'])

    # 计算平均指标
    mean_auc = np.nanmean(auc_scores)
    mean_accuracy = np.nanmean(accuracy_scores)
    mean_precision = np.nanmean(precision_scores)
    mean_recall = np.nanmean(recall_scores)
    mean_f1 = np.nanmean(f1_scores)
    mean_fpr = np.nanmean(fpr_scores)

    results = {
        "scores": scores.tolist(),
        "targets": targets.tolist(),
        "predictions": binary_scores.tolist(),
        "ids": [i.tolist() for i in ids],
        # 每个类别的指标
        "per_class_metrics": {
            "auc": auc_scores,
            "accuracy": accuracy_scores,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores,
            "false_positive_rate": fpr_scores
        },
        # 平均指标
        "mean_metrics": {
            "mean_auc": mean_auc,
            "mean_accuracy": mean_accuracy,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1_score": mean_f1,
            "mean_false_positive_rate": mean_fpr
        }
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_classifier(config, args.test_csv, args.checkpoint, args.device)
    test_set_name = args.test_csv.split("/")[-1:][0]

    with open(args.checkpoint[:-4] + f"results_{test_set_name}.json", "w") as f:
        json.dump(results, f)
