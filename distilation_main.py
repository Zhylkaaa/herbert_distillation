from typing import List
from argparse import ArgumentParser
from functools import partial

import torch
import evaluate
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import load_dataset

from monkey_patches import HerbertDataCollatorForWholeWordMask
from losses import DistillationLoss


def tokenize_and_split(examples, tokenizer, max_length=256):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
    )


class DistilTrainer(Trainer):
    def __init__(self, teacher_model, student_model, loss_fn, temperature=None, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature if temperature else 1.
        self.loss_fn = loss_fn

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        mask = inputs['labels'] != -100
        masked_student_logits = student_output.logits[mask]  # B x vocab
        masked_teacher_logits = teacher_output.logits[mask]  # B x vocab
        masked_labels = inputs['labels'][mask]

        loss, mlm_loss, *_ = self.loss_fn(masked_student_logits, masked_teacher_logits, masked_labels, return_parts=True)
        with torch.no_grad():
            self.log({'perplexity': torch.exp(mlm_loss).item()})

        return (loss, student_output) if return_outputs else loss


if __name__ == '__main__':
    teacher_model = AutoModelForMaskedLM.from_pretrained('allegro/herbert-large-cased')
    student_config = AutoConfig.from_pretrained('allegro/herbert-base-cased')
    student_config._name_or_path = 'Zhylkaaa/distil-herbert-cased'
    student_config.num_hidden_layers = student_config.num_hidden_layers // 2

    student_model = AutoModelForMaskedLM.from_config(student_config)
    tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-large-cased')

    dataset = load_dataset('distillation_corpus')

    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=256)
    tokenized_dataset_256 = dataset.map(tokenize, batched=True, num_proc=16)

    data_collator = HerbertDataCollatorForWholeWordMask(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='distil_bert_out',
        fp16=True,
        fsdp='full_shard',
        report_to='wandb',
        learning_rate=5e-5,
        warmup_ratio=0.06,
        num_train_epochs=3,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=21,
        per_device_eval_batch_size=21,
        gradient_accumulation_steps=1,
        evaluation_strategy='steps',
        eval_steps=0.5,
        save_strategy='steps',
        save_steps=0.2
    )

    loss_fn = DistillationLoss(temperature=2.)

    trainer = DistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        loss_fn=loss_fn,
        training_args=training_args,
        train_dataset=tokenized_dataset_256["train"],
        eval_dataset=tokenized_dataset_256["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train()

    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=512)
    tokenized_dataset_512 = dataset.map(tokenize, batched=True, num_proc=16)

    training_args.max_steps = 10000

    trainer = DistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        loss_fn=loss_fn,
        training_args=training_args,
        train_dataset=tokenized_dataset_512["train"],
        eval_dataset=tokenized_dataset_512["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train()

