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


class DistilModel(torch.nn.Module):
    def __init__(self, student_model: torch.nn.Module, teacher_model: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    def forward(self, x):
        student_output = self.student_model(**x)
        teacher_output = self.teacher_model(**x)
        return student_output, teacher_output

    def train(self, mode: bool = True):
        self.student_model.train(mode)
        return self

    def parameters(self, recurse: bool = True):
        return self.student_model.parameters(recurse=recurse)

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ):
        return self.student_model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)


class DistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, loss_fn=None, temperature=None, *args, **kwargs):
        model = DistilModel(teacher_model, student_model)
        super().__init__(model=model, *args, **kwargs)
        self.temperature = temperature if temperature else 1.
        self.loss_fn = loss_fn
        self.perplexity = torch.tensor(0.)

    def compute_loss(self, model, inputs, return_outputs=False):
        student_output, teacher_output = model(inputs)

        mask = inputs['labels'] != -100
        masked_student_logits = student_output.logits[mask]  # B x vocab
        masked_teacher_logits = teacher_output.logits[mask]  # B x vocab
        masked_labels = inputs['labels'][mask]

        loss, mlm_loss, *_ = self.loss_fn(masked_student_logits, masked_teacher_logits, masked_labels, return_parts=True)

        perp = torch.exp(mlm_loss).item()
        self.perplexity += perp

        if self.control.should_log:
            perplexity_scalar = self._nested_gather(self.perplexity).mean().item()
            perplexity_scalar = perplexity_scalar / (self.state.global_step - self._globalstep_last_logged)
            self.log({'perplexity': perplexity_scalar})

        return (loss, student_output) if return_outputs else loss

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.perplexity = torch.tensor(0.).to(args.device)
        super()._inner_training_loop(batch_size=batch_size,
                                     args=args,
                                     resume_from_checkpoint=resume_from_checkpoint,
                                     trial=trial,
                                     ignore_keys_for_eval=ignore_keys_for_eval)


if __name__ == '__main__':
    teacher_model = AutoModelForMaskedLM.from_pretrained('allegro/herbert-large-cased')
    student_config = AutoConfig.from_pretrained('allegro/herbert-base-cased')
    student_config._name_or_path = 'Zhylkaaa/distil-herbert-cased'
    student_config.num_hidden_layers = student_config.num_hidden_layers // 2

    student_model = AutoModelForMaskedLM.from_config(student_config)
    tokenizer = AutoTokenizer.from_pretrained('allegro/herbert-large-cased')

    dataset = load_dataset('distillation_corpus')

    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=256)
    tokenized_dataset_256 = dataset.map(tokenize,
                                        batched=True,
                                        num_proc=16,
                                        remove_columns=dataset['train'].column_names)
    tokenized_dataset_256.set_format('torch', columns=["input_ids", "token_type_ids", "attention_mask"])

    data_collator = HerbertDataCollatorForWholeWordMask(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='distil_bert_out',
        report_to='wandb',
        learning_rate=5e-5,
        warmup_ratio=0.06,
        num_train_epochs=3,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        evaluation_strategy='steps',
        eval_steps=0.5,
        save_strategy='steps',
        save_steps=0.2,
        remove_unused_columns=False,
    )

    loss_fn = DistillationLoss(temperature=2.)

    trainer = DistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        loss_fn=loss_fn,
        args=training_args,
        train_dataset=tokenized_dataset_256["train"],
        eval_dataset=tokenized_dataset_256["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train()

    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=512)
    tokenized_dataset_512 = dataset.map(tokenize,
                                        batched=True,
                                        num_proc=16,
                                        remove_columns=dataset['train'].column_names)

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

