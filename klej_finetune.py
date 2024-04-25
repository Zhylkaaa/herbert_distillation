from argparse import ArgumentParser
from functools import partial

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torch import nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy, Precision, Recall, F1Score

from model_utils import get_linear_schedule_with_warmup
from klej_datamodule import KlejDatamodule


METRICS = {
    'accuracy': partial(Accuracy, task='multiclass'),
    'precision': partial(Precision, task='multiclass'),
    'recall': partial(Recall, task='multiclass'),
    'f1': partial(F1Score, task='multiclass')
}


class KlejTrainer(L.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay, warmup_steps):
        super().__init__()
        self.num_classes = num_classes
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.loss_fn = nn.CrossEntropyLoss()

        self._init_metrics()

    def _init_metrics(self):
        # TODO: separate train and test metrics?
        self.train_metrics = {
            k: metric(num_classes=self.num_classes) for k, metric in METRICS.items()
        }
        self.val_metrics = {
            k: metric(num_classes=self.num_classes) for k, metric in METRICS.items()
        }
        self.test_metrics = {
            k: metric(num_classes=self.num_classes) for k, metric in METRICS.items()
        }

        for prefix, metrics in (('train', self.train_metrics),
                                ('val', self.val_metrics),
                                ('test', self.test_metrics)):
            [self.register_module(f'{prefix}_{k}', v) for k, v in metrics.items()]

    def forward(self, batch):
        return self.model(**batch).logits

    def _log_metrics(self, pred, y, metrics, stage='', on_step=True, on_epoch=True):
        for metric_name, metric in metrics.items():
            metric(pred, y)
            self.log(f'{stage}_{metric_name}', metric, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def _common_step(self, batch, stage, metrics):
        preds = self(batch)
        labels = batch['labels']

        loss = self.loss_fn(preds, labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        preds = torch.softmax(preds.detach(), dim=1)
        self._log_metrics(preds,
                          labels,
                          metrics,
                          stage)
        return loss

    def training_step(self, batch, batch_id):
        return self._common_step(batch, 'train', self.train_metrics)

    def validation_step(self, batch, batch_id):
        return self._common_step(batch, 'val', self.val_metrics)

    def test_step(self, batch, batch_id):
        return self._common_step(batch, 'test', self.test_metrics)

    def predict_step(self, batch, batch_id):
        return self(batch)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        decay_params = [
            p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        if isinstance(self.warmup_steps, float):
            warmup_steps = int(num_training_steps * self.warmup_steps)
        else:
            warmup_steps = self.warmup_steps

        lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=num_training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": 'step',
                "frequency": 1,
                "strict": True,
            }
        }


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='distil_herbert_out', type=str)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--data_path', default='klej_data/klej_polemo2.0-in/', type=str)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--warmup_steps', default=0.06, type=float)
    parser.add_argument('--monitor_metric', default='val_accuracy_epoch', type=str)
    parser.add_argument('--prefix', default='model_preds', type=str)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project='distil_herbert_finetune')
    checkpoints = ModelCheckpoint(
        monitor=args.monitor_metric,
        verbose=True,
        save_top_k=1,
        mode='max'
    )
    callbacks = [LearningRateMonitor(), checkpoints]

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        gradient_clip_val=2.0,
        accumulate_grad_batches=1,
        devices=1,
        accelerator='gpu',
        strategy='ddp',
        precision='16-mixed',
        callbacks=callbacks,
        logger=wandb_logger
    )

    warmup_steps = args.warmup_steps if args.warmup_steps < 1. else int(args.warmup_steps)

    datamodule = KlejDatamodule(data_path=args.data_path,
                                max_length=args.max_length,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size)

    model = KlejTrainer(
        model_name=args.model_name,
        num_classes=len(datamodule.label_names),
        weight_decay=args.weight_decay,
        lr=args.lr,
        warmup_steps=warmup_steps
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule, ckpt_path='best')
    out = trainer.predict(model, datamodule=datamodule, ckpt_path='best')
    torch.save(out, args.prefix + '_' + args.data_path.split('/')[1] + '_preds.pt')
