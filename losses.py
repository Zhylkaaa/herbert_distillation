import torch
from torch import nn
from torch.nn import functional as F

from similarity_measures import LinearMeasure


class DistillationLoss(nn.Module):
    def __init__(
            self,
            target_lambda=0.3,
            kl_lambda=0.3,
            similarity_lambda=0.3,
            temperature=2.,
            similarity_measure=None,
            **similarity_measure_kwargs
    ):
        super().__init__()
        self.register_buffer('target_lambda', torch.tensor(target_lambda))
        self.register_buffer('kl_lambda', torch.tensor(kl_lambda))
        self.register_buffer('similarity_lambda', torch.tensor(similarity_lambda))
        self.register_buffer('temperature', torch.tensor(temperature))

        self.mlm_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.similarity_measure = similarity_measure
        if similarity_measure == 'cosine':
            self.similarity_loss = nn.CosineEmbeddingLoss()
        elif similarity_measure == 'linear':
            self.similarity_loss = LinearMeasure(**similarity_measure_kwargs)
        elif similarity_measure is None or similarity_measure == 'none':
            self.similarity_loss = None
        else:
            raise ValueError(f'Unrecognized similarity measure {similarity_measure}')

    def forward(self,
                student_logits,
                teacher_logits,
                targets,
                student_hidden=None,
                teacher_hidden=None,
                return_parts=False):
        if not self.training:
            return self.mlm_loss(student_logits, targets)
        mlm_loss = self.mlm_loss(student_logits, targets)

        soft_log_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = self.kl_div(soft_log_student, soft_teacher)

        output = self.target_lambda * mlm_loss + self.kl_lambda * self.temperature ** 2 * kl_loss

        sim_loss = None
        if not (self.similarity_lambda < 1e-8 or self.similarity_loss is None):
            if self.similarity_measure == 'cosine':
                sim_loss = self.similarity_loss(student_hidden,
                                                teacher_hidden,
                                                torch.ones(student_hidden.shape[:-1],
                                                           device=student_hidden.device).long())
            elif self.similarity_measure == 'linear':
                sim_loss = self.similarity_loss(student_hidden.unsqueeze(1),
                                                teacher_hidden.unsqueeze(1))
            sim_loss = sim_loss

        if sim_loss is not None:
            output += self.similarity_lambda * sim_loss
        if return_parts:
            output = (output, mlm_loss, kl_loss) + ((sim_loss,) if sim_loss else ())
        return output
