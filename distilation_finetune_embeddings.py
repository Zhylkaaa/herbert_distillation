from argparse import ArgumentParser
from functools import partial

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

from distilation_main import DistilTrainer, tokenize_and_split
from monkey_patches import HerbertDataCollatorForWholeWordMask
from losses import DistillationLoss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--student_model', required=True)
    parser.add_argument('--teacher_model', default='allegro/herbert-large-cased')
    parser.add_argument('--dataset_name', default='distillation_corpus')
    parser.add_argument('--num_proc', default=16, type=int)
    parser.add_argument('--output_dir', default='distil_herbert_out')
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--fine_tuning_steps', default=10000, type=int)
    parser.add_argument('--per_device_train_batch_size', default=32, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
    parser.add_argument('--dataloader_num_workers', default=10, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eval_steps', default=0.5, type=float)
    parser.add_argument('--save_steps', default=0.2, type=float)
    parser.add_argument('--target_lambda', default=0.5, type=float)
    parser.add_argument('--kl_lambda', default=0.5, type=float)
    parser.add_argument('--cosine_lambda', default=0., type=float)
    parser.add_argument('--temperature', default=2., type=float)
    args = parser.parse_args()

    teacher_model = AutoModelForMaskedLM.from_pretrained(args.teacher_model)
    student_model = AutoModelForMaskedLM.from_pretrained(args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    dataset = load_dataset(args.dataset_name)

    data_collator = HerbertDataCollatorForWholeWordMask(tokenizer=tokenizer)
    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=512)
    tokenized_dataset_512 = dataset.map(tokenize,
                                        batched=True,
                                        num_proc=args.num_proc,
                                        remove_columns=dataset['train'].column_names)
    tokenized_dataset_512.set_format('torch', columns=["input_ids", "token_type_ids", "attention_mask"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to='wandb',
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.fine_tuning_steps,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=True
    )

    loss_fn = DistillationLoss(target_lambda=args.target_lambda,
                               kl_lambda=args.kl_lambda,
                               cosine_lambda=args.cosine_lambda,
                               temperature=args.temperature)

    trainer = DistilTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        loss_fn=loss_fn,
        args=training_args,
        train_dataset=tokenized_dataset_512["train"],
        eval_dataset=tokenized_dataset_512["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train()
