import torch
from torch import Tensor
from torch.nn.functional import pad

from typing import Literal, Tuple, Optional, List


class LinearMeasure(torch.nn.Module):
    def __init__(self,
                 alpha=1, center_columns=True, dim_matching='zero_pad', svd_grad=True, reduction='mean'):
        super(LinearMeasure, self).__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        assert dim_matching in [None, 'none', 'zero_pad', 'pca']
        self.dim_matching = dim_matching
        self.center_columns = center_columns
        self.svd_grad = svd_grad
        self.reduction = reduction

    def partial_fit(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the mean centered columns. Can be replaced later by whitening transform for linear invarariances."""
        if self.center_columns:
            mx = torch.mean(X, dim=1, keepdim=True)
        else:
            mx = torch.zeros(X.shape[2], dtype=X.dtype, device=X.device)
        wx = X - mx
        return mx, wx

    def fit(self, X: Tensor, Y: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        mx, wx = self.partial_fit(X)
        my, wy = self.partial_fit(Y)

        if self.svd_grad:
            wxy = torch.bmm(wx.transpose(1, 2), wy)
            U, _, Vt = torch.linalg.svd(wxy)
        else:
            with torch.no_grad():
                wxy = torch.bmm(wx.transpose(1, 2), wy)
                U, _, Vt = torch.linalg.svd(wxy)
        wx = U
        wy = Vt.transpose(1, 2)
        return (mx, wx), (my, wy)

    def project(self, X: Tensor, m: Tensor, w: Tensor):
        if self.center_columns:
            return torch.bmm((X - m), w)
        else:
            return torch.bmm(X, w)

    def forward(self, X: Tensor, Y: Tensor):
        if X.shape[:-1] != Y.shape[:-1] or X.ndim != 3 or Y.ndim != 3:
            raise ValueError('Expected 3D input matrices to much in all dimensions but last.'
                             f'But got {X.shape} and {Y.shape} instead.')

        if X.shape[-1] != Y.shape[-1]:
            if self.dim_matching is None or self.dim_matching == 'none':
                raise ValueError(f'Expected same dimension matrices got instead {X.shape} and {Y.shape}. '
                                 f'Set dim_matching or change matrix dimensions.')
            elif self.dim_matching == 'zero_pad':
                size_diff = Y.shape[-1] - X.shape[-1]
                if size_diff < 0:
                    raise ValueError(f'With `zero_pad` dimension matching expected X dimension to be smaller then Y. '
                                     f'But got {X.shape} and {Y.shape} instead.')
                X = pad(X, (0, size_diff))
            elif self.dim_matching == 'pca':
                raise NotImplementedError
            else:
                raise ValueError(f'Unrecognized dimension matching {self.reduction}')

        X_params, Y_params = self.fit(X, Y)
        norms = torch.linalg.norm(self.project(X, *X_params) - self.project(Y, *Y_params), ord="fro", dim=(1, 2))

        if self.reduction == 'mean':
            return norms.mean()
        elif self.reduction == 'sum':
            return norms.sum()
        elif self.reduction == 'none' or self.reduction is None:
            return norms
        else:
            raise ValueError(f'Unrecognized reduction {self.reduction}')
