from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    BertModel,
    BertForMaskedLM,
)
from transformers.models.bert.modeling_bert import BertEncoder
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


# credit: https://github.com/BartekKrzepkowski/DistilHerBERT-base_vol2/blob/master/models/distil_student.py
def copy_weights_to_student(init_model, student):
    if isinstance(init_model, BertModel) or isinstance(init_model, BertForMaskedLM):
        for teacher_part, student_part in zip(init_model.children(), student.children()):
            copy_weights_to_student(teacher_part, student_part)
    elif isinstance(init_model, BertEncoder):
        teacher_encoding_layers = [layer for layer in init_model.layer.children()][::2]
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[i].state_dict())
    else:
        student.load_state_dict(init_model.state_dict())


def get_student_model(student_model_name):
    initialization_model = AutoModelForMaskedLM.from_pretrained(student_model_name)
    student_config = AutoConfig.from_pretrained(student_model_name)
    student_config._name_or_path = 'Zhylkaaa/distil-herbert-cased'
    student_config.num_hidden_layers = student_config.num_hidden_layers // 2

    student_model = AutoModelForMaskedLM.from_config(student_config)

    copy_weights_to_student(initialization_model, student_model)
    return student_model


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
