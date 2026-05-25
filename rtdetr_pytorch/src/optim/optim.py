
import torch 
import torch.nn as nn 
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR', 'WarmupCosineLR']


class WarmupCosineLR(lr_scheduler._LRScheduler):
	def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 0.0, last_epoch: int = -1):
		self.warmup_epochs = max(0, int(warmup_epochs))
		self.total_epochs = max(int(total_epochs), self.warmup_epochs + 1)
		self.min_lr = float(min_lr)
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
			warmup_factor = (self.last_epoch + 1) / float(self.warmup_epochs)
			return [base_lr * warmup_factor for base_lr in self.base_lrs]

		progress = (self.last_epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
		cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
		return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
LambdaLR = register(lr_scheduler.LambdaLR)
WarmupCosineLR = register(WarmupCosineLR)
