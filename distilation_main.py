import os
from typing import List, Optional
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
from transformers.trainer import TRAINING_ARGS_NAME, unwrap_model, logger
from datasets import load_dataset

from monkey_patches import HerbertDataCollatorForWholeWordMask
from losses import DistillationLoss
from model_utils import get_student_model


def tokenize_and_split(examples, tokenizer, max_length=256):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
    )


class DistilModel(torch.nn.Module):
    _keys_to_ignore_on_save = set()

    def __init__(self, student_model: torch.nn.Module, teacher_model: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        DistilModel._keys_to_ignore_on_save = set(['teacher_model.'+k for k in self.teacher_model.state_dict()])

    def forward(self, x, **kwargs):
        if 'return_loss' in x:
            del x['return_loss']

        student_output = self.student_model(**x, **kwargs)
        # absolutely unnecessary, but gives me peace of mind
        with torch.no_grad():
            teacher_output = self.teacher_model(**x, **kwargs)

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
    def __init__(self, student_model=None, teacher_model=None, loss_fn=None, temperature=None, *args, **kwargs):
        model = DistilModel(student_model, teacher_model)
        super().__init__(model=model, *args, **kwargs)
        self.temperature = temperature if temperature else 1.
        self.loss_fn = loss_fn
        self.perplexity = torch.tensor(0.)

    def compute_loss(self, model, inputs, return_outputs=False):
        student_output, teacher_output = model(inputs, output_hidden_states=True)

        mask = inputs['labels'] != -100

        masked_student_logits = student_output.logits[mask]  # B x vocab

        B, L, h = student_output.hidden_states[-1].shape
        n_layers = len(student_output.hidden_states) - 1
        masked_student_states = torch.stack(student_output.hidden_states[1:]).view(B * L, n_layers, h)
        masked_student_states = masked_student_states[mask.view(-1)]

        masked_teacher_logits = teacher_output.logits[mask]  # B x vocab

        B, L, h = teacher_output.hidden_states[-1].shape
        n_layers = len(teacher_output.hidden_states) - 1
        masked_teacher_states = torch.stack(teacher_output.hidden_states[1:]).view(B * L, n_layers, h)
        masked_teacher_states = masked_teacher_states[mask.view(-1)]

        masked_labels = inputs['labels'][mask]

        loss, mlm_loss, *_ = self.loss_fn(masked_student_logits,
                                          masked_teacher_logits,
                                          masked_labels,
                                          student_hidden=masked_student_states,
                                          teacher_hidden=masked_teacher_states,
                                          return_parts=True)

        if self.model.training:
            perp = torch.exp(mlm_loss).item()
            self.perplexity += perp

        return (loss, student_output) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs['return_loss'] = True
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.perplexity = torch.tensor(0.).to(args.device)
        super()._inner_training_loop(batch_size=batch_size,
                                     args=args,
                                     resume_from_checkpoint=resume_from_checkpoint,
                                     trial=trial,
                                     ignore_keys_for_eval=ignore_keys_for_eval)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            perplexity_scalar = self._nested_gather(self.perplexity).mean().item()
            perplexity_scalar = perplexity_scalar / (self.state.global_step - self._globalstep_last_logged)
            self.log({'perplexity': perplexity_scalar})
            self.perplexity -= self.perplexity
            self.control.should_log = True

        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = unwrap_model(self.model).student_model
        state_dict = {k[len('student_model.'):]: v for k, v in state_dict.items() if k.startswith('student_model')} \
            if state_dict else None

        model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--teacher_model', default='allegro/herbert-large-cased')
    parser.add_argument('--student_model', default='allegro/herbert-base-cased')
    parser.add_argument('--dataset_name', default='distillation_corpus')
    parser.add_argument('--num_proc', default=16, type=int)
    parser.add_argument('--output_dir', default='distil_herbert_out')
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.06, type=float)
    parser.add_argument('--num_train_epochs', default=3, type=int)
    parser.add_argument('--per_device_train_batch_size', default=32, type=int)
    parser.add_argument('--per_device_eval_batch_size', default=32, type=int)
    parser.add_argument('--dataloader_num_workers', default=10, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eval_steps', default=0.5, type=float)
    parser.add_argument('--save_steps', default=0.2, type=float)
    parser.add_argument('--target_lambda', default=0.5, type=float)
    parser.add_argument('--kl_lambda', default=0.5, type=float)
    parser.add_argument('--similarity_lambda', default=0., type=float)
    parser.add_argument('--similarity_measure', default=None, choices=['none', 'cosine', 'linear'])
    parser.add_argument('--no_center_columns', action='store_true')
    parser.add_argument('--full_similarity', action='store_true')
    parser.add_argument('--no_svd_grad', action='store_true')
    parser.add_argument('--dim_matching', default='zero_pad', choices=['zero_pad', 'none'])
    parser.add_argument('--temperature', default=2., type=float)
    parser.add_argument('--resume_from_checkpoint', default=None, type=str)
    args = parser.parse_args()
    teacher_model = AutoModelForMaskedLM.from_pretrained(args.teacher_model)
    student_model = get_student_model(args.student_model)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    dataset = load_dataset(args.dataset_name)

    tokenize = partial(tokenize_and_split, tokenizer=tokenizer, max_length=256)
    tokenized_dataset_256 = dataset.map(tokenize,
                                        batched=True,
                                        num_proc=args.num_proc,
                                        remove_columns=dataset['train'].column_names)
    tokenized_dataset_256.set_format('torch', columns=["input_ids", "token_type_ids", "attention_mask"])

    data_collator = HerbertDataCollatorForWholeWordMask(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to='wandb',
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
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
                               similarity_lambda=args.similarity_lambda,
                               temperature=args.temperature,
                               similarity_measure=args.similarity_measure,
                               full_similarity=args.full_similarity,
                               center_columns=not args.no_center_columns,
                               svd_grad=not args.no_svd_grad,
                               dim_matching=args.dim_matching)

    trainer = DistilTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        loss_fn=loss_fn,
        args=training_args,
        train_dataset=tokenized_dataset_256["train"],
        eval_dataset=tokenized_dataset_256["validation"],
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
