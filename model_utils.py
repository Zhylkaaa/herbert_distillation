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
