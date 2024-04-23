import os
from functools import partial

import lightning as L
import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


def tokenize_and_split(examples, tokenizer, max_length=256):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
    )


class KlejDatamodule(L.LightningDataModule):
    def __init__(self, data_path, tokenizer_name='allegro/herbert-large-cased', max_length=256, num_workers=3, batch_size=32):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.num_workers = num_workers
        if self.num_workers > 0:
            self.mp_context = 'spawn'
        else:
            self.mp_context = None
        self.batch_size = batch_size

        train_dataset = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t', quoting=3, skip_blank_lines=False)
        train_dataset['label'] = train_dataset.target.astype('category')
        label_dtype = train_dataset.label.dtype

        if os.path.exists(os.path.join(data_path, 'dev.tsv')):
            path = 'dev.tsv'
        else:
            path = 'train.tsv'
        val_dataset = pd.read_csv(os.path.join(data_path, path), sep='\t', quoting=3, skip_blank_lines=False)
        val_dataset['label'] = val_dataset.target.astype(label_dtype)

        self.label_names = label_dtype.categories

        train_dataset['label'] = train_dataset['label'].cat.codes
        val_dataset['label'] = val_dataset['label'].cat.codes

        self.data = DatasetDict({
            'train': Dataset.from_pandas(train_dataset),
            'val': Dataset.from_pandas(val_dataset),
            'test': Dataset.from_pandas(val_dataset),
        })

        tokenize = partial(tokenize_and_split, tokenizer=self.tokenizer, max_length=self.max_length)
        self.data = self.data.map(tokenize)
        self.data.set_format('torch', columns=["input_ids", "token_type_ids", "attention_mask", 'label'])

        # Prediction dataset does not have labels
        pred_dataset = pd.read_csv(os.path.join(data_path, 'test_features.tsv'), sep='\t', quoting=3, skip_blank_lines=False)
        self.pred_dataset = Dataset.from_pandas(pred_dataset)
        self.pred_dataset = self.pred_dataset.map(tokenize)
        self.pred_dataset.set_format('torch', columns=["input_ids", "token_type_ids", "attention_mask"])

    def train_dataloader(self):
        return DataLoader(
            self.data['train'],
            num_workers=self.num_workers,
            multiprocessing_context=self.mp_context,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.data['val'],
            num_workers=self.num_workers,
            multiprocessing_context=self.mp_context,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data['test'],
            num_workers=self.num_workers,
            multiprocessing_context=self.mp_context,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            num_workers=self.num_workers,
            multiprocessing_context=self.mp_context,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False
        )
