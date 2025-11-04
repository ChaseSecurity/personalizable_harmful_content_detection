# flake8: noqa: E402
import argparse
import json
import os

# Disable tokenizer parallelism to avoid deadlocks due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import src.data_loaders as module_data
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader


class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges."""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = config["model"]["num_classes"]
        self.model_args = {
            "model_type": config["model"]["model_type"],
            "model_name": config["model"]["model_name"],
            "tokenizer_name": config["model"]["tokenizer_name"],
            "num_classes": config["model"]["num_classes"]
        }
        self.model, self.tokenizer = get_model_and_tokenizer(** self.model_args)
        self.bias_loss = False

        if "loss_weight" in config:
            self.loss_weight = config["loss_weight"]
        if "num_main_classes" in config:
            self.num_main_classes = config["num_main_classes"]
            self.bias_loss = True
        else:
            self.num_main_classes = self.num_classes

        self.config = config
        self.val_outputs = []

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.val_outputs.append({"loss": loss, "acc": acc})
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def on_validation_epoch_end(self):
        val_loss_list = [output["loss"] for output in self.val_outputs]
        val_acc_list = [output["acc"] for output in self.val_outputs]
        
        avg_val_loss = torch.stack(val_loss_list).mean().cpu().item()
        avg_val_acc = torch.stack(val_acc_list).mean().cpu().item()
        
        # 关键修改1：记录平均验证准确率，供ModelCheckpoint监控
        self.log("avg_val_acc", avg_val_acc, prog_bar=True)
        self.log("avg_val_loss", avg_val_loss, prog_bar=False)
        
        print(f"\n=====================================")
        print(f"Epoch {self.current_epoch + 1:2d} | Average Val Loss: {avg_val_loss:.6f}")
        print(f"Epoch {self.current_epoch + 1:2d} | Average Val Acc:  {avg_val_acc:.6f}")
        print(f"=====================================\n")

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        loss = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

    def binary_cross_entropy(self, input, meta):
        """Custom binary_cross_entropy function."""
        if "weight" in meta:
            target = meta["target"].to(input.device).reshape(input.shape)
            weight = meta["weight"].to(input.device).reshape(input.shape)
            return F.binary_cross_entropy_with_logits(input, target, weight=weight)
        elif "multi_target" in meta:
            target = meta["multi_target"].to(input.device)
            loss_fn = F.binary_cross_entropy_with_logits
            mask = target != -1
            loss = loss_fn(input, target.float(), reduction="none")

            if "class_weights" in meta:
                weights = meta["class_weights"][0].to(input.device)
            elif "weights1" in meta:
                weights = meta["weights1"].to(input.device)
            else:
                weights = torch.tensor(1 / self.num_main_classes).to(input.device)
                loss = loss[:, : self.num_main_classes]
                mask = mask[:, : self.num_main_classes]

            weighted_loss = loss * weights
            nz = torch.sum(mask, 0) != 0
            masked_tensor = weighted_loss * mask
            masked_loss = torch.sum(masked_tensor[:, nz], 0) / torch.sum(mask[:, nz], 0)
            loss = torch.sum(masked_loss)
            return loss
        else:
            target = meta["target"].to(input.device)
            return F.binary_cross_entropy_with_logits(input, target.float())

    def binary_accuracy(self, output, meta):
        """Custom binary_accuracy function."""
        if "multi_target" in meta:
            target = meta["multi_target"].to(output.device)
        else:
            target = meta["target"].to(output.device)
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0
        return torch.tensor(correct)


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, type=str, help="config file path (default: None)"
    )
    parser.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    parser.add_argument(
        "-d", "--device", default=None, type=str, help="comma-separated indices of GPUs to enable (default: None)"
    )
    parser.add_argument(
        "--num_workers", default=10, type=int, help="number of workers used in the data loader (default: 10)"
    )
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="if given, override the num")

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # 配置新的保存目录
    new_save_root = f"/root/autodl-tmp/detoxify/saved/{config['name']}"
    os.makedirs(new_save_root, exist_ok=True)

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    # model
    model = ToxicClassifier(config)

    # 关键修改2：将检查点回调改为监控平均验证准确率，模式改为最大化
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        verbose=True,
        monitor="avg_val_acc",  # 改为监控平均验证准确率
        mode="max",  # 准确率越高越好，所以模式改为max
    )

    if args.device is None:
        devices = "auto"
    else:
        devices = [int(d.strip()) for d in args.device.split(",")]

    trainer = pl.Trainer(
        devices=devices,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        default_root_dir=new_save_root,
        deterministic=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_loader,
        val_dataloaders=valid_data_loader,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    cli_main()
