import torch
from torch import Tensor
from torchmetrics import Metric


class RVDMetric(Metric):
    def __init__(self, num_outputs: int, eps: float = 1e-2, **kwargs):
        """
        Args:
            num_outputs (int): The number of ouputs (i.e., number of features per sample).
            eps (float): A small value to avoid division by zero.
        """
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.eps = eps
        
        self.add_state("sum_pred", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_pred_sq", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_gt", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_gt_sq", default=torch.zeros(num_outputs), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Update the metric state with a new batch of predictions and targets.
        Both tensors should be of shape (N, num_outputs), where N is the batch size.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        if preds.shape[1] != self.num_outputs:
            raise ValueError(f"Expected predictions with {self.num_outputs} outputs, got {preds.shape[1]} instead.")
        
        batch_size = preds.shape[0]
        
        self.sum_pred += preds.sum(dim=0)
        self.sum_pred_sq += (preds ** 2).sum(dim=0)
        self.sum_gt += target.sum(dim=0)
        self.sum_gt_sq += (target ** 2).sum(dim=0)
        self.n += batch_size

    def compute(self) -> Tensor:
        """
        Compute the Relative Variation Distance (RVD).
        
        RVD = (1 / C) * sum_{i=1}^C {[(variance_pred_i - variance_gt_i) / (variance_gt_i + eps)]^2}
        
        where the variances are computed as the population variance.
        """
        # Total count of samples
        n = self.n
        
        # Compute the population variance for each gene.
        var_pred = self.sum_pred_sq / n - (self.sum_pred / n) ** 2
        var_gt = self.sum_gt_sq / n - (self.sum_gt / n) ** 2

        # Compute the relative difference per gene and its squared value.
        relative_diff = (var_pred - var_gt) / (var_gt + self.eps)
        rvd = torch.mean(relative_diff ** 2)
        
        return rvd