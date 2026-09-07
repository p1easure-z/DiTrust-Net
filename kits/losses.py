import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_loss(loss_name, **kwargs):
    """Factory to initialize loss functions."""
    loss_map = {
        'BinaryCrossEntropy': BinaryCrossEntropy,
        'LogCoshDiceLoss': LogCoshDiceLoss,
    }
    if loss_name not in loss_map:
        raise NotImplementedError(f"Loss '{loss_name}' not recognized.")

    return loss_map[loss_name](**kwargs)


# --- Loss Implementations ---

class BinaryCrossEntropy(nn.Module):
    """Positive-weighted BCE on probabilities for BraTS multi-label targets."""
    def __init__(self, reduction="mean", pos_weight=(1.0, 1.3, 1.6), eps=1e-6, from_logits=False):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.from_logits = from_logits
        self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))

    def forward(self, output, target):
        target = target.float()

        if output.dim() == 4:
            weight = self.pos_weight.view(1, -1, 1, 1)
        else:
            weight = self.pos_weight

        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
            loss = loss * (1.0 + (weight - 1.0) * target)
        else:
            output = output.clamp(self.eps, 1.0 - self.eps)
            loss = -(weight * target * torch.log(output) + (1.0 - target) * torch.log(1.0 - output))

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class LogCoshDiceLoss(nn.Module):
    """Class-aware log-cosh Dice to avoid flattening all BraTS labels together."""
    def __init__(self, smooth=1.0, class_weights=(1.0, 1.3, 1.6), from_logits=False):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, input, target):
        target = target.to(device=input.device, dtype=input.dtype)
        if self.from_logits:
            input = torch.sigmoid(input)

        if input.dim() != 4:
            iflat = input.reshape(-1)
            tflat = target.reshape(-1)
            intersection = (iflat * tflat).sum()
            dice_score = (2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
            dice_loss = 1 - dice_score
            return torch.log(torch.cosh(dice_loss))

        intersection = (input * target).sum(dim=(0, 2, 3))
        union = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        class_weights = self.class_weights[:dice_loss.shape[0]].to(device=input.device, dtype=input.dtype)
        class_weights = class_weights / class_weights.sum()

        dice_loss = (dice_loss * class_weights).sum()
        return torch.log(torch.cosh(dice_loss))
