import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_utils import gather_tensor, get_rank


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        z_i: InChI projections [Batch, Dim]
        z_j: SMILES projections [Batch, Dim]
        """
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        logits = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(batch_size).to(z_i.device)

        loss_i2j = self.criterion(logits, labels)
        loss_j2i = self.criterion(logits.T, labels)

        return (loss_i2j + loss_j2i) / 2


class ReconstructionLoss(nn.Module):


    def __init__(self, ignore_index=0):
        super(ReconstructionLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred_logits, target_seq):
        """
        pred_logits: [Batch, Seq_Len-1, Vocab_Size]
        target_seq:  [Batch, Seq_Len-1]
        """
        pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))
        target_flat = target_seq.reshape(-1)

        return self.criterion(pred_flat, target_flat)