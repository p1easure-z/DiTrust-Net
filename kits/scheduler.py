import math


# class LR_Scheduler(object):
#     """
#     Learning Rate Scheduler.
#     Modes: 'cos' (Cosine Annealing), 'poly', 'step'.
#     """
#
#     def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
#         self.mode = mode
#         self.lr = base_lr
#         self.lr_step = lr_step
#         self.iters_per_epoch = iters_per_epoch
#         self.N = num_epochs * iters_per_epoch
#         self.epoch = -1
#         self.warmup_iters = warmup_epochs * iters_per_epoch
#
#     def __call__(self, optimizer, i, epoch, best_dice_pre, best_dice_epoch, best_iou_pre, best_iou_epoch):
#         T = epoch * self.iters_per_epoch + i
#
#         # Calculate LR based on mode
#         if self.mode == 'cos':
#             lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
#         elif self.mode == 'poly':
#             lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
#         elif self.mode == 'step':
#             lr = self.lr * (0.1 ** (epoch // self.lr_step))
#         else:
#             raise NotImplementedError(f"Scheduler mode {self.mode} not implemented")
#
#         # Warmup logic
#         if self.warmup_iters > 0 and T < self.warmup_iters:
#             lr = lr * 1.0 * T / self.warmup_iters
#
#         # Logging only when epoch changes
#         if epoch > self.epoch:
#             self.epoch = epoch
#
#         self._adjust_learning_rate(optimizer, lr)
#
#     def _adjust_learning_rate(self, optimizer, lr):
#         if len(optimizer.param_groups) == 1:
#             optimizer.param_groups[0]['lr'] = lr
#         else:
#             # enlarge the lr at the head
#             optimizer.param_groups[0]['lr'] = lr
#             for i in range(1, len(optimizer.param_groups)):
#                 optimizer.param_groups[i]['lr'] = lr * 10

import math
import torch


class LR_Scheduler(object):
    """
    [v2.0 Upgrade] Learning Rate Scheduler with Decoupled Warmup & Min_LR.
    Modes: 'cos' (Cosine Annealing), 'poly', 'step'.
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0, min_lr=1e-6):
        self.mode = mode
        self.base_lr = base_lr
        self.min_lr = min_lr  # [新增] 防止最后学习率变为0
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = num_epochs * iters_per_epoch

        # 确保 Cosine 计算时的周期不包含 warmup 部分
        self.decay_iters = self.total_iters - self.warmup_iters

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i

        # --- Phase 1: Warmup ---
        if self.warmup_iters > 0 and T < self.warmup_iters:
            # 从 0 线性增加到 base_lr
            lr = self.base_lr * (T / self.warmup_iters)

        # --- Phase 2: Decay ---
        else:
            # 计算已经度过了多少 decay 时间
            T_decay = T - self.warmup_iters

            if self.mode == 'cos':
                # 标准 Cosine 公式 (带 min_lr)
                # 0.5 * (max - min) * (1 + cos) + min
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                        1 + math.cos(math.pi * T_decay / self.decay_iters))

            elif self.mode == 'poly':
                # Poly 策略通常不加 min_lr，直接衰减到 0
                lr = self.base_lr * pow((1 - 1.0 * T_decay / self.decay_iters), 0.9)

            elif self.mode == 'step':
                lr = self.base_lr * (0.1 ** (epoch // self.lr_step))

            else:
                raise NotImplementedError(f"Scheduler mode {self.mode} not implemented")

        # Logging only when epoch changes
        if epoch > self.epoch:
            self.epoch = epoch

        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        # 如果有多个 param_groups，后面的学习率放大 10 倍
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            # group[0] 通常是 backbone, group[1+] 是 head/decoder
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10