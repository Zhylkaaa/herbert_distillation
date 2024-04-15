import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    def __init__(self, target_lambda=0.3, kl_lambda=0.3, cosine_lambda=0.3, temperature=2.):
        super().__init__()
        self.register_buffer('target_lambda', torch.tensor(target_lambda))
        self.register_buffer('kl_lambda', torch.tensor(kl_lambda))
        self.register_buffer('cosine_lambda', torch.tensor(cosine_lambda))
        self.register_buffer('temperature', torch.tensor(temperature))

        self.mlm_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.similarity_loss = nn.CosineEmbeddingLoss()

    def forward(self, student_logits, teacher_logits, targets, return_parts=False):
        if not self.training:
            return self.mlm_loss(student_logits, targets)
        mlm_loss = self.target_lambda * self.mlm_loss(student_logits, targets)

        soft_log_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl_loss = self.kl_lambda * self.temperature ** 2 * self.kl_div(soft_log_student, soft_teacher)

        # # should be loss between hidden states (which are not the same size, so just disable)
        # sim_loss = self.cosine_lambda * self.similarity_loss(student_last_hidden,
        #                                                      teacher_last_hidden,
        #                                                      torch.ones_like(student_last_hidden))

        # output = mlm_loss + kl_loss + sim_loss
        output = mlm_loss + kl_loss
        if return_parts:
            output = (output, mlm_loss, kl_loss)
        return output
