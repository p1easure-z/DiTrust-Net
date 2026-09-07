import warnings
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backbone.vmamba.vmamba import SS2D, vmamba_tiny_s1l8
from kits.utils import print_state_dict_match_report, summarize_state_dict_match

warnings.filterwarnings("ignore")

BACKBONE_DIMS = {
    "vmamba": [96, 192, 384, 768],
}


def normalize_backbone_name(backbone_name):
    name = str(backbone_name or "vmamba").lower()
    if name in {"vmamba", "vssm", "vmamba_tiny", "vmamba_tiny_s1l8"}:
        return "vmamba"
    raise ValueError(f"Unsupported backbone '{backbone_name}'. Available: vmamba.")


def build_backbone(backbone_name):
    normalize_backbone_name(backbone_name)
    return vmamba_tiny_s1l8()


def get_backbone_dims(backbone_name):
    return list(BACKBONE_DIMS[normalize_backbone_name(backbone_name)])


def default_backbone_weight_path(project_root, backbone_name):
    normalize_backbone_name(backbone_name)
    return os.path.join(project_root, "backbone", "vmamba", "ckpt", "vssm1_tiny_0230s_ckpt_epoch_264.pth")


def _extract_state_dict(weight_payload):
    if isinstance(weight_payload, dict):
        for key in ("model", "state_dict", "net"):
            value = weight_payload.get(key)
            if isinstance(value, dict):
                return value
    return weight_payload


def _strip_common_prefixes(key):
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def _normalize_vmamba_state_dict(state_dict):
    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = _strip_common_prefixes(key)
        if ".ln_1." in new_key:
            new_key = new_key.replace(".ln_1.", ".norm.")
        if ".ln_2." in new_key:
            new_key = new_key.replace(".ln_2.", ".norm2.")
        if ".self_attention." in new_key:
            new_key = new_key.replace(".self_attention.", ".op.")
        if "patch_embed.proj." in new_key:
            new_key = new_key.replace("patch_embed.proj.", "patch_embed.0.")
        if "patch_embed.norm." in new_key:
            new_key = new_key.replace("patch_embed.norm.", "patch_embed.2.")
        normalized[new_key] = value
    return normalized


def load_pretrained_backbone_pair(core_backbone, whole_backbone, backbone_name, weight_path):
    backbone_name = normalize_backbone_name(backbone_name)
    weight_payload = torch.load(weight_path, map_location="cpu")
    state_dict = _extract_state_dict(weight_payload)
    normalized_state = _normalize_vmamba_state_dict(state_dict)

    matched_state, report = summarize_state_dict_match(core_backbone, normalized_state)
    print_state_dict_match_report(f"{backbone_name} backbone pretrained", weight_path, report)
    if report["matched_keys"] == 0:
        raise RuntimeError(f"No compatible {backbone_name} backbone keys were found in '{weight_path}'")

    core_backbone.load_state_dict(matched_state, strict=False)
    whole_backbone.load_state_dict(matched_state, strict=False)
    return report


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GateBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        gate = torch.sigmoid(self.gate_conv(x))
        x_val = self.act(self.in_proj(x))
        return self.out_proj(x_val * gate) + residual


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        active_gates = torch.isfinite(gates) & (gates > 0)
        nonzero_gates = torch.nonzero(active_gates, as_tuple=False)

        if nonzero_gates.numel() == 0:
            batch_size = gates.size(0)
            device = gates.device
            self._expert_index = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            self._batch_index = torch.arange(batch_size, device=device)
            self._part_sizes = [batch_size] + [0] * (self._num_experts - 1)
            self._nonzero_gates = torch.ones((batch_size, 1), device=device, dtype=gates.dtype)
            return

        sorted_experts, index_sorted_experts = nonzero_gates.sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = nonzero_gates[index_sorted_experts[:, 1], 0]
        self._part_sizes = active_gates.sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1).unsqueeze(1))
        zeros = torch.zeros(
            (self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3)),
            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.silu(x + residual)


class BandExpert(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, role, band_mask, kernel_size=3, dilation=1):
        super().__init__()
        self.role = role
        self.register_buffer("band_mask", band_mask)
        self.band_scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        padding = ((kernel_size - 1) // 2) * dilation
        self.band_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.decoder = DecoderLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def _resolve_mask(self, x, dynamic_mask=None):
        mask = dynamic_mask if dynamic_mask is not None else self.band_mask
        if mask.size(0) == 1 and x.size(0) > 1:
            mask = mask.expand(x.size(0), -1, -1, -1)
        if mask.size(1) == 1 and x.size(1) > 1:
            mask = mask.expand(mask.size(0), x.size(1), mask.size(2), mask.size(3))
        return mask.to(dtype=x.dtype, device=x.device)

    def forward(self, x, dynamic_mask=None):
        x = self._sanitize_tensor(x)
        mask = self._resolve_mask(x, dynamic_mask=dynamic_mask)
        spec = torch.fft.rfft2(x, norm='ortho')
        band_spec = torch.complex(spec.real * mask, spec.imag * mask)
        band_feat = torch.fft.irfft2(band_spec, s=x.shape[-2:], norm='ortho')
        band_feat = self.band_proj(self._sanitize_tensor(band_feat))
        x = x + torch.tanh(self.band_scale) * band_feat
        return self.decoder(x)


class CDMoE(nn.Module):
    """Complementary Dirichlet Mixture-of-Experts router."""
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        num_experts,
        hidden_channels,
        noisy_gating=True,
        k=2,
        selector_capacity_weight=0.02,
        expert_balance_weight=0.03,
        capacity_exploration_mass=0.08,
    ):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k = k
        self.freq_channels = 4 * in_channels
        self.noise_expert_index = num_experts - 1
        self.register_buffer(
            "selection_temperature",
            torch.tensor(1.0, dtype=torch.float32),
        )
        self.dirichlet_concentration_scale = 0.5
        self.alpha_eps = 1e-4
        self.alpha_margin = 1e-3
        self.selection_sparsity_weight = 0.01
        self.capacity_consistency_weight = 0.03
        self.selector_capacity_weight = float(selector_capacity_weight)
        self.expert_balance_weight = float(expert_balance_weight)
        self.capacity_log_eps = 1e-6
        self.capacity_exploration_mass = float(capacity_exploration_mass)
        self.soft_routing_warmup = False

        low_mask, mid_mask, high_mask = self._build_band_masks(img_size, img_size // 2 + 1)
        self.register_buffer("low_band_mask", low_mask)
        self.register_buffer("mid_band_mask", mid_mask)
        self.register_buffer("high_band_mask", high_mask)

        self.prompt_encoder = nn.Sequential(
            nn.Conv2d(self.freq_channels, in_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )
        self.prompt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.token_to_prompt = nn.Sequential(
            nn.Linear(8, in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, in_channels),
        )
        router_input = 8 + 2 * in_channels
        self.router_context = nn.Sequential(
            nn.Linear(router_input, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.selection_head = nn.Linear(hidden_channels, num_experts)
        nn.init.normal_(self.selection_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.selection_head.bias)
        self.alpha_lo_head = nn.Linear(hidden_channels, num_experts)
        self.alpha_delta_head = nn.Linear(hidden_channels, num_experts)
        self.band_experts = nn.ModuleList([
            BandExpert(
                in_channels, out_channels, hidden_channels,
                role="low", band_mask=self.low_band_mask, kernel_size=5, dilation=1,
            ),
            BandExpert(
                in_channels, out_channels, hidden_channels,
                role="mid", band_mask=self.mid_band_mask, kernel_size=3, dilation=1,
            ),
            BandExpert(
                in_channels, out_channels, hidden_channels,
                role="high", band_mask=self.high_band_mask, kernel_size=3, dilation=2,
            ),
            BandExpert(
                in_channels, out_channels, hidden_channels,
                role="noise", band_mask=self.high_band_mask, kernel_size=3, dilation=1,
            ),
        ])
        self.water_level_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        initial_capacity_scale = 0.5
        self.capacity_logit_scale_raw = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(initial_capacity_scale, dtype=torch.float32)))
        )

        self.softplus = nn.Softplus()
        assert self.k <= self.num_experts

    def set_routing_schedule(
        self,
        epoch,
        total_epochs,
        warmup_epochs=5,
        temperature_start=2.0,
        temperature_end=0.35,
    ):
        total_epochs = max(int(total_epochs), 1)
        warmup_epochs = min(max(int(warmup_epochs), 0), total_epochs)
        temperature_start = max(float(temperature_start), 1e-3)
        temperature_end = max(float(temperature_end), 1e-3)

        if epoch < warmup_epochs:
            warmup_progress = float(epoch) / max(warmup_epochs - 1, 1)
            temperature = temperature_start + warmup_progress * (1.0 - temperature_start)
            self.soft_routing_warmup = True
        else:
            anneal_progress = float(epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
            anneal_progress = min(max(anneal_progress, 0.0), 1.0)
            temperature = 1.0 + anneal_progress * (temperature_end - 1.0)
            self.soft_routing_warmup = False

        self.selection_temperature.fill_(max(float(temperature), 1e-3))

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def _safe_normalize(self, values, dim=1, eps=1e-6):
        values = self._sanitize_tensor(values, posinf=1e4, neginf=0.0).clamp_min(0.0)
        denom = values.sum(dim=dim, keepdim=True)
        fallback = torch.full_like(values, 1.0 / values.size(dim))
        invalid = (~torch.isfinite(denom)) | (denom <= eps)
        return torch.where(invalid.expand_as(values), fallback, values / denom.clamp_min(eps))

    def _relative_capacity_allocation(self, eta_values, eps=1e-6):
        eta_values = self._sanitize_tensor(eta_values, posinf=1e4, neginf=0.0).clamp_min(0.0)
        work_eta = eta_values.float() if eta_values.dtype in (torch.float16, torch.bfloat16) else eta_values
        eta_mean = work_eta.mean(dim=1, keepdim=True)
        positive_mean = torch.isfinite(eta_mean) & (eta_mean > 0.0)
        normalization_floor = torch.finfo(work_eta.dtype).tiny
        relative_eta = torch.where(
            positive_mean,
            work_eta / eta_mean.clamp_min(normalization_floor),
            torch.zeros_like(work_eta),
        ).to(dtype=eta_values.dtype)

        water_level = 1.0 + torch.sigmoid(self.water_level_raw)
        water_level = water_level.to(dtype=eta_values.dtype, device=eta_values.device).view(1, 1)
        inverse_cost = 1.0 / (relative_eta + eps)
        raw_allocation = F.relu(water_level - inverse_cost)
        allocation = self._safe_normalize(raw_allocation, dim=1, eps=eps)
        return allocation

    @staticmethod
    def _build_band_masks(height, width):
        freq_y = torch.fft.fftfreq(height, d=1.0)
        freq_x = torch.fft.rfftfreq((width - 1) * 2, d=1.0)
        grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        radius = torch.sqrt(grid_y.square() + grid_x.square())
        radius = radius / radius.max().clamp_min(1e-6)
        low = (radius <= 0.30).float().unsqueeze(0).unsqueeze(0)
        mid = ((radius > 0.30) & (radius <= 0.65)).float().unsqueeze(0).unsqueeze(0)
        high = (radius > 0.65).float().unsqueeze(0).unsqueeze(0)
        return low, mid, high

    @staticmethod
    def _expand_mask(mask, tensor):
        if mask.size(0) == 1 and tensor.size(0) > 1:
            mask = mask.expand(tensor.size(0), -1, -1, -1)
        if mask.size(1) == 1 and tensor.size(1) > 1:
            mask = mask.expand(mask.size(0), tensor.size(1), mask.size(2), mask.size(3))
        return mask.to(dtype=tensor.dtype, device=tensor.device)

    def _masked_mean(self, tensor, mask):
        mask = self._expand_mask(mask, tensor)
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        return (tensor * mask).sum(dim=(1, 2, 3)) / denom

    def _masked_complex_mean(self, tensor, mask):
        mask = self._expand_mask(mask, tensor.real)
        denom = mask.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        real = (tensor.real * mask).sum(dim=(1, 2, 3)) / denom
        imag = (tensor.imag * mask).sum(dim=(1, 2, 3)) / denom
        return torch.complex(real, imag)

    def _local_coherence(self, cross_spec, power_x, power_r, eps=1e-6):
        cross_real = F.avg_pool2d(cross_spec.real, kernel_size=3, stride=1, padding=1)
        cross_imag = F.avg_pool2d(cross_spec.imag, kernel_size=3, stride=1, padding=1)
        local_power_x = F.avg_pool2d(power_x, kernel_size=3, stride=1, padding=1)
        local_power_r = F.avg_pool2d(power_r, kernel_size=3, stride=1, padding=1)
        coherence = (cross_real.square() + cross_imag.square()) / (local_power_x * local_power_r + eps)
        return self._sanitize_tensor(coherence, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def capacity_prompt(self, x, ref):
        eps = 1e-6
        x = self._sanitize_tensor(x)
        ref = self._sanitize_tensor(ref)
        spec_x = torch.fft.rfft2(x, norm='ortho')
        spec_r = torch.fft.rfft2(ref, norm='ortho')
        raw_cross_spec = spec_x * torch.conj(spec_r)
        cross_spec = torch.complex(
            self._sanitize_tensor(raw_cross_spec.real, posinf=1e6, neginf=-1e6),
            self._sanitize_tensor(raw_cross_spec.imag, posinf=1e6, neginf=-1e6),
        )
        raw_cross_power = cross_spec / cross_spec.abs().clamp_min(eps)
        cross_power = torch.complex(
            self._sanitize_tensor(raw_cross_power.real, posinf=1.0, neginf=-1.0),
            self._sanitize_tensor(raw_cross_power.imag, posinf=1.0, neginf=-1.0),
        )
        power_x = self._sanitize_tensor(spec_x.abs().square(), posinf=1e6, neginf=0.0)
        power_r = self._sanitize_tensor(spec_r.abs().square(), posinf=1e6, neginf=0.0)
        coherence_map = self._local_coherence(cross_spec, power_x, power_r, eps=eps)
        noise_mask = (1.0 - coherence_map.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)

        band_masks = [self.low_band_mask, self.mid_band_mask, self.high_band_mask, noise_mask]
        eta_values = []
        capacity_values = []
        for mask in band_masks:
            cross_mean = self._masked_complex_mean(cross_spec, mask)
            px_mean = self._masked_mean(power_x, mask)
            pr_mean = self._masked_mean(power_r, mask)
            gamma_sq = (cross_mean.abs().square() / (px_mean * pr_mean + eps)).clamp(0.0, 1.0 - 1e-4)
            eta = gamma_sq / (1.0 - gamma_sq + eps)
            eta_values.append(eta)
            capacity_values.append(torch.log1p(eta))

        eta_values = torch.stack(eta_values, dim=1)
        capacity_values = torch.stack(capacity_values, dim=1)
        waterfill_allocation = self._relative_capacity_allocation(eta_values, eps=eps)
        exploration_mass = min(max(float(self.capacity_exploration_mass), 0.0), 1.0)
        allocation = (
            (1.0 - exploration_mass) * waterfill_allocation +
            exploration_mass / self.num_experts
        )
        band_tokens = torch.cat([capacity_values, allocation], dim=1)

        freq_feature = torch.cat([
            self._sanitize_tensor((power_x + eps).sqrt()),
            self._sanitize_tensor((power_r + eps).sqrt()),
            self._sanitize_tensor(cross_power.real, posinf=1.0, neginf=-1.0),
            self._sanitize_tensor(cross_power.imag, posinf=1.0, neginf=-1.0),
        ], dim=1)
        prompt_context = self.prompt_pool(self.prompt_encoder(freq_feature))
        token_prompt = self.token_to_prompt(band_tokens).unsqueeze(-1).unsqueeze(-1)
        fre_prompt = torch.sigmoid(prompt_context + token_prompt)

        return fre_prompt, allocation, band_tokens, noise_mask

    @staticmethod
    def _gap(x):
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

    def _sample_logistic_noise(self, reference):
        uniform = torch.rand_like(reference).clamp_(1e-6, 1.0 - 1e-6)
        return torch.log(uniform) - torch.log1p(-uniform)

    def _relaxed_bernoulli_selection(self, selection_logits):
        if self.training and self.noisy_gating and not self.soft_routing_warmup:
            selection_logits = selection_logits + self._sample_logistic_noise(selection_logits)
        temperature = self.selection_temperature.to(
            device=selection_logits.device,
            dtype=selection_logits.dtype,
        ).clamp_min(1e-6)
        selection_z = torch.sigmoid(selection_logits / temperature)
        return selection_z

    def _capacity_adjusted_logits(self, context_logits, allocation):
        log_capacity = torch.log(allocation.clamp_min(self.capacity_log_eps))
        centered_log_capacity = log_capacity - log_capacity.mean(dim=1, keepdim=True)
        capacity_logit_scale = F.softplus(self.capacity_logit_scale_raw)
        adjusted_logits = context_logits + capacity_logit_scale * centered_log_capacity
        return self._sanitize_tensor(adjusted_logits)

    def _dirichlet_contribution(self, alpha):
        dirichlet_mean = self._safe_normalize(alpha, dim=1)
        if not self.training or self.soft_routing_warmup:
            return dirichlet_mean, dirichlet_mean

        sample = Dirichlet(alpha.float()).rsample().to(dtype=alpha.dtype, device=alpha.device)
        dirichlet_sample = self._safe_normalize(sample, dim=1)
        return dirichlet_sample, dirichlet_mean

    def _dirichlet_routing(self, x_enhanced, ref, band_tokens, allocation):
        context = torch.cat([band_tokens, self._gap(x_enhanced), self._gap(ref)], dim=1)
        context = self.router_context(self._sanitize_tensor(context))

        context_selection_logits = self._sanitize_tensor(self.selection_head(context))
        selection_logits = self._capacity_adjusted_logits(
            context_selection_logits,
            allocation,
        )
        temperature = self.selection_temperature.to(
            device=selection_logits.device,
            dtype=selection_logits.dtype,
        ).clamp_min(1e-6)
        selection_distribution = F.softmax(selection_logits / temperature, dim=1)
        selection_z = self._relaxed_bernoulli_selection(selection_logits)

        alpha_lo = self.softplus(self.alpha_lo_head(context)) + self.alpha_eps
        alpha_hi = alpha_lo + self.softplus(self.alpha_delta_head(context)) + self.alpha_margin
        alpha = self.dirichlet_concentration_scale * (selection_z * alpha_hi + (1.0 - selection_z) * alpha_lo)
        alpha = self._sanitize_tensor(alpha, posinf=1e4, neginf=self.alpha_eps).clamp_min(self.alpha_eps)
        expert_contribution, dirichlet_mean = self._dirichlet_contribution(alpha)
        routing_weight = self._safe_normalize(
            selection_distribution * selection_z * expert_contribution,
            dim=1,
        )

        capacity_target = allocation.detach()
        capacity_loss = (
            capacity_target * (
                torch.log(capacity_target.clamp_min(1e-6)) -
                torch.log(dirichlet_mean.clamp_min(1e-6))
            )
        ).sum(dim=1).mean()
        selector_capacity_loss = (
            capacity_target * (
                torch.log(capacity_target.clamp_min(self.capacity_log_eps)) -
                torch.log(selection_distribution.clamp_min(self.capacity_log_eps))
            )
        ).sum(dim=1).mean()

        batch_expert_usage = selection_distribution.mean(dim=0)
        uniform_expert_usage = torch.full_like(batch_expert_usage, 1.0 / self.num_experts)
        soft_importance_balance_loss = (
            batch_expert_usage * (
                torch.log(batch_expert_usage.clamp_min(self.capacity_log_eps)) -
                torch.log(uniform_expert_usage)
            )
        ).sum()

        selector_topk_indices = selection_distribution.topk(
            min(self.k, self.num_experts),
            dim=1,
        ).indices
        hard_selection = torch.zeros_like(selection_distribution)
        hard_selection.scatter_(1, selector_topk_indices, 1.0)
        soft_multi_selection = float(self.k) * selection_distribution
        straight_through_selection = (
            hard_selection + soft_multi_selection - soft_multi_selection.detach()
        )
        batch_expert_hard_load = straight_through_selection.mean(dim=0) / float(self.k)
        hard_load_balance_loss = (
            batch_expert_hard_load - uniform_expert_usage
        ).square().sum()
        expert_balance_loss = soft_importance_balance_loss + hard_load_balance_loss

        sparsity_loss = (selection_z.sum(dim=1) - float(self.k)).square().mean()
        sparsity_term = self.selection_sparsity_weight * sparsity_loss
        capacity_term = self.capacity_consistency_weight * capacity_loss
        selector_capacity_term = self.selector_capacity_weight * selector_capacity_loss
        balance_term = self.expert_balance_weight * expert_balance_loss
        cdmoe_loss = sparsity_term + capacity_term + selector_capacity_term + balance_term
        cdmoe_loss = torch.nan_to_num(cdmoe_loss, nan=0.0, posinf=1.0, neginf=0.0)

        loss_terms = {
            "sparsity_term": sparsity_term,
            "capacity_term": capacity_term,
            "selector_capacity_term": selector_capacity_term,
            "balance_term": balance_term,
        }
        return routing_weight, cdmoe_loss, selection_distribution, loss_terms

    def _topk_gates(self, routing_weight, selection_score):
        k = min(self.k, self.num_experts)
        top_indices = selection_score.topk(k, dim=1).indices
        top_values = self._sanitize_tensor(
            routing_weight.gather(1, top_indices),
            posinf=1e4,
            neginf=0.0,
        ).clamp_min(1e-6)
        selected_sum = top_values.sum(dim=1, keepdim=True)
        top_values = top_values / selected_sum
        gates = torch.zeros_like(routing_weight)
        gates.scatter_(1, top_indices, top_values)
        return gates

    def _run_sparse_experts(self, x_enhanced, noise_mask, gates):
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_enhanced)
        noise_masks = dispatcher.dispatch(noise_mask)
        expert_outputs = []
        for i, expert_input in enumerate(expert_inputs):
            if expert_input.size(0) == 0:
                output_shape = (0, self.out_channels, x_enhanced.size(2), x_enhanced.size(3))
                expert_outputs.append(x_enhanced.new_zeros(output_shape))
                continue
            dynamic_mask = noise_masks[i] if i == self.noise_expert_index else None
            expert_outputs.append(self.band_experts[i](expert_input, dynamic_mask=dynamic_mask))
        return self._sanitize_tensor(dispatcher.combine(expert_outputs))

    def forward(self, x, ref):
        x = self._sanitize_tensor(x)
        ref = self._sanitize_tensor(ref)
        fre_prompt, allocation, band_tokens, noise_mask = self.capacity_prompt(x, ref)
        x_enhanced = self._sanitize_tensor(x * fre_prompt + x)
        routing_weight, cdmoe_loss, selection_distribution, loss_terms = self._dirichlet_routing(
            x_enhanced=x_enhanced,
            ref=ref,
            band_tokens=band_tokens,
            allocation=allocation,
        )

        hard_dispatch_gates = self._topk_gates(
            routing_weight,
            selection_score=selection_distribution,
        )
        if self.training and self.soft_routing_warmup:
            dispatch_gates = routing_weight
        elif self.training:
            dispatch_gates = hard_dispatch_gates + routing_weight - routing_weight.detach()
        else:
            dispatch_gates = hard_dispatch_gates
        y = self._run_sparse_experts(x_enhanced, noise_mask, dispatch_gates)

        return y, cdmoe_loss, loss_terms


def _pait_group_norm(num_channels, max_groups=32):
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class BranchDisagreementEstimator(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(dim * 2 + 1, hidden, kernel_size=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
        )

        map_hidden = max(dim // 4, 16)
        self.disagreement_head = nn.Sequential(
            nn.Conv2d(dim + 1, map_hidden, kernel_size=3, padding=1, bias=False),
            _pait_group_norm(map_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(map_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.disagreement_head[-2].bias, -1.5)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, x1, x2):
        x1 = self._sanitize_tensor(x1)
        x2 = self._sanitize_tensor(x2)
        signed_diff = self._sanitize_tensor(x2 - x1)
        abs_diff = signed_diff.abs()
        cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-6).unsqueeze(1)
        cos_dist = ((1.0 - cos_sim) * 0.5).clamp(0.0, 1.0)

        innovation_context = self.diff_encoder(torch.cat([abs_diff, signed_diff, cos_dist], dim=1))
        disagreement_map = self.disagreement_head(torch.cat([innovation_context, cos_dist], dim=1))
        return disagreement_map, innovation_context


class PriorStructuralEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 2, 16)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _sanitize_tensor(x, posinf=1.0, neginf=0.0):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    @staticmethod
    def _spatial_boundary(prior_map):
        if prior_map.size(-2) > 1:
            grad_h = (prior_map[:, :, 1:, :] - prior_map[:, :, :-1, :]).abs()
            grad_h = F.pad(grad_h, (0, 0, 0, 1), mode="replicate")
        else:
            grad_h = torch.zeros_like(prior_map)

        if prior_map.size(-1) > 1:
            grad_w = (prior_map[:, :, :, 1:] - prior_map[:, :, :, :-1]).abs()
            grad_w = F.pad(grad_w, (0, 1, 0, 0), mode="replicate")
        else:
            grad_w = torch.zeros_like(prior_map)

        boundary = (grad_h + grad_w).sum(dim=1, keepdim=True)
        boundary = boundary / boundary.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return boundary.clamp(0.0, 1.0)

    def _prepare_prior(self, prior_map, x_ref):
        if prior_map is None:
            b, _, h, w = x_ref.shape
            return x_ref.new_zeros((b, 3, h, w))

        if prior_map.dim() != 4:
            raise ValueError(f"prior_map must be a 4D tensor, but got shape {tuple(prior_map.shape)}.")
        if prior_map.size(0) != x_ref.size(0):
            raise ValueError(
                "prior_map and feature batch size must match, "
                f"but got {prior_map.size(0)} and {x_ref.size(0)}."
            )
        if prior_map.size(1) != 3:
            raise ValueError(
                "prior_map must have 3 channels in WT/TC/ET order, "
                f"but got {prior_map.size(1)}."
            )

        prior_map = self._sanitize_tensor(prior_map).clamp(0.0, 1.0)
        if prior_map.size(-2) != x_ref.size(-2) or prior_map.size(-1) != x_ref.size(-1):
            prior_map = F.interpolate(prior_map, size=x_ref.shape[-2:], mode="bilinear", align_corners=False)

        return prior_map

    def forward(self, prior_map, x_ref):
        prior_map = self._prepare_prior(prior_map, x_ref)
        prior_feat = self.prior_encoder(prior_map)
        prior_boundary = self._spatial_boundary(prior_map)
        region_cues = prior_map
        return prior_feat, prior_boundary, region_cues


class StructuralStateEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cue_channels = max(dim // 4, 16)
        self.structure_embed = nn.Sequential(
            nn.Conv2d(1, cue_channels, kernel_size=1, bias=False),
            _pait_group_norm(cue_channels),
            nn.SiLU(inplace=True),
        )

        self.branch_norm = _pait_group_norm(dim)

        self.innovation_context = nn.Sequential(
            nn.Conv2d(dim + cue_channels, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
        )

        self.prior_context = nn.Sequential(
            nn.Conv2d(dim + cue_channels, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
        )

        self.state_encoder = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, x1, x2, innovation_context, structure_cue, prior_feat):
        structure_feat = self.structure_embed(structure_cue)
        branch_context = F.silu(self.branch_norm(0.5 * (x1 + x2)))
        innovation_context = self.innovation_context(torch.cat([innovation_context, structure_feat], dim=1))
        prior_context = self.prior_context(torch.cat([prior_feat, structure_feat], dim=1))
        state_input = torch.cat([branch_context, innovation_context, prior_context], dim=1)
        return self.state_encoder(self._sanitize_tensor(state_input))


class AnisotropicDirectionField(nn.Module):
    def __init__(self, dim, num_directions=8):
        super().__init__()
        self.num_directions = num_directions
        hidden = max(dim // 4, 16)
        self.direction_head = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_directions, kernel_size=1, bias=True),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.direction_head[-1].bias)
        nn.init.constant_(self.confidence_head[-2].bias, -1.5)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, structural_state):
        direction_logits = self._sanitize_tensor(self.direction_head(structural_state)).clamp(-30.0, 30.0)
        direction_weights = F.softmax(direction_logits, dim=1)
        direction_weights = torch.nan_to_num(
            direction_weights,
            nan=1.0 / self.num_directions,
            posinf=1.0 / self.num_directions,
            neginf=0.0,
        )
        direction_weights = direction_weights / direction_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        direction_confidence = self.confidence_head(structural_state).clamp(0.0, 1.0)
        return direction_logits, direction_weights, direction_confidence


class DirectionalInnovationTransport(nn.Module):
    def __init__(self, dim, num_directions=8):
        super().__init__()
        if num_directions != 8:
            raise ValueError("DirectionalInnovationTransport requires 8 directions for paired opposite routing.")

        base_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        opposite_indices = [4, 5, 6, 7, 0, 1, 2, 3]
        self.direction_offsets = base_offsets
        self.register_buffer("opposite_indices", torch.tensor(opposite_indices, dtype=torch.long), persistent=False)
        self.transport_strength = nn.Parameter(torch.tensor(-2.0))
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    @staticmethod
    def _shift_with_replicate_padding(x, dy, dx):
        h, w = x.shape[-2:]
        pad_top = max(-dy, 0)
        pad_bottom = max(dy, 0)
        pad_left = max(-dx, 0)
        pad_right = max(dx, 0)
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        start_y = max(dy, 0)
        start_x = max(dx, 0)
        return padded[:, :, start_y:start_y + h, start_x:start_x + w]

    def forward(self, innovation, direction_weights, direction_confidence):
        innovation = self._sanitize_tensor(innovation)
        flux = torch.zeros_like(innovation)
        for direction_idx, (dy, dx) in enumerate(self.direction_offsets):
            shifted = self._shift_with_replicate_padding(innovation, dy, dx)
            directional_flux = self._sanitize_tensor(shifted - innovation)
            direction_weight = direction_weights[:, direction_idx:direction_idx + 1]
            flux = flux + direction_weight * directional_flux

        transport_scale = 0.25 * torch.sigmoid(self.transport_strength)
        transported = innovation + transport_scale * direction_confidence * flux
        transported = self._sanitize_tensor(transported)
        projected = self.output_proj(transported)
        return self._sanitize_tensor(transported + projected)


class TransmissionController(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = max(dim // 4, 16)
        region_channels = 8
        interaction_channels = 8
        self.region_embed = nn.Sequential(
            nn.Conv2d(3, region_channels, kernel_size=1, bias=False),
            _pait_group_norm(region_channels),
            nn.SiLU(inplace=True),
        )
        self.interaction_embed = nn.Sequential(
            nn.Conv2d(3, interaction_channels, kernel_size=1, bias=False),
            _pait_group_norm(interaction_channels),
            nn.SiLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Conv2d(dim + region_channels + interaction_channels, hidden, kernel_size=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
        )
        self.transmission_head = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        self.reception_head = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        self.commit_head = nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        nn.init.constant_(self.transmission_head.bias, -2.0)
        nn.init.constant_(self.reception_head.bias, -2.0)
        nn.init.constant_(self.commit_head.bias, -1.0)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, structural_state, structure_cue, region_cues, transported_12, transported_21):
        transport_mag_12 = transported_12.abs().mean(dim=1, keepdim=True)
        transport_mag_21 = transported_21.abs().mean(dim=1, keepdim=True)
        region_feat = self.region_embed(region_cues)
        interaction_feat = self.interaction_embed(torch.cat([
            structure_cue,
            transport_mag_12,
            transport_mag_21,
        ], dim=1))
        control_input = torch.cat([structural_state, region_feat, interaction_feat], dim=1)
        control_feat = self.shared(self._sanitize_tensor(control_input))
        return {
            "transmission": torch.sigmoid(self.transmission_head(control_feat)),
            "reception": torch.sigmoid(self.reception_head(control_feat)),
            "commit": torch.sigmoid(self.commit_head(control_feat)),
        }


class GlobalStructuralRefiner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.merge_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
            nn.GELU(),
        )
        self.proj_norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(
            d_model=dim, d_state=16, ssm_ratio=2.0, dt_rank="auto",
            act_layer=nn.SiLU, d_conv=3, conv_bias=True, forward_type="v2",
            dropout=0.0, initialize="v0"
        )
        self.delta_x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
        )
        self.delta_x2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            _pait_group_norm(dim),
        )

        hidden = max(dim // 4, 16)
        self.context_gate = nn.Sequential(
            nn.Conv2d(dim + 1, hidden, kernel_size=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.context_gate[-2].bias, -2.0)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, aligned_x1, aligned_x2, structure_cue):
        fused = self.merge_proj(torch.cat([aligned_x1, aligned_x2], dim=1))
        x_in = self.proj_norm(fused.permute(0, 2, 3, 1).contiguous())
        global_context = self.ss2d(x_in).permute(0, 3, 1, 2).contiguous()
        global_context = self._sanitize_tensor(global_context + fused)

        gate_input = torch.cat([global_context, structure_cue], dim=1)
        gate_x1, gate_x2 = self.context_gate(gate_input).chunk(2, dim=1)
        out_x1 = aligned_x1 + gate_x1 * self.delta_x1(global_context)
        out_x2 = aligned_x2 + gate_x2 * self.delta_x2(global_context)
        return self._sanitize_tensor(out_x1), self._sanitize_tensor(out_x2)


class PriorGuidedAnisotropicInnovationTransport(nn.Module):
    def __init__(self, dim, num_directions=8):
        super().__init__()
        self.disagreement_estimator = BranchDisagreementEstimator(dim=dim)
        self.prior_encoder = PriorStructuralEncoder(dim=dim)
        self.state_encoder = StructuralStateEncoder(dim=dim)
        self.direction_field = AnisotropicDirectionField(dim=dim, num_directions=num_directions)
        self.transport = DirectionalInnovationTransport(dim=dim, num_directions=num_directions)
        self.controller = TransmissionController(dim=dim)
        self.global_refiner = GlobalStructuralRefiner(dim=dim)

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    def forward(self, x1, x2, prior_map=None):
        if x1.shape != x2.shape:
            raise ValueError(f"x1 and x2 must have the same shape, got {tuple(x1.shape)} and {tuple(x2.shape)}.")
        if x1.dim() != 4:
            raise ValueError(f"x1 and x2 must be 4D tensors, but got shape {tuple(x1.shape)}.")

        x1 = self._sanitize_tensor(x1)
        x2 = self._sanitize_tensor(x2)

        prior_feat, prior_boundary, region_cues = self.prior_encoder(prior_map, x1)
        disagreement_map, innovation_context = self.disagreement_estimator(x1, x2)
        structure_cue = torch.maximum(disagreement_map, prior_boundary).clamp(0.0, 1.0)
        structural_state = self.state_encoder(
            x1=x1,
            x2=x2,
            innovation_context=innovation_context,
            structure_cue=structure_cue,
            prior_feat=prior_feat,
        )

        _, direction_weights, direction_confidence = self.direction_field(structural_state)
        opposite_indices = self.transport.opposite_indices.to(direction_weights.device)
        direction_weights_21 = direction_weights.index_select(dim=1, index=opposite_indices)
        innovation_12 = self._sanitize_tensor(x2 - x1)
        innovation_21 = self._sanitize_tensor(x1 - x2)
        transported_12 = self.transport(innovation_12, direction_weights, direction_confidence)
        transported_21 = self.transport(innovation_21, direction_weights_21, direction_confidence)

        controls = self.controller(
            structural_state=structural_state,
            structure_cue=structure_cue,
            region_cues=region_cues,
            transported_12=transported_12,
            transported_21=transported_21,
        )
        transmit_12, transmit_21 = controls["transmission"].chunk(2, dim=1)
        receive_1, receive_2 = controls["reception"].chunk(2, dim=1)
        commit_1, commit_2 = controls["commit"].chunk(2, dim=1)

        update_x1 = transmit_12 * receive_1 * commit_1 * transported_12
        update_x2 = transmit_21 * receive_2 * commit_2 * transported_21
        aligned_x1 = self._sanitize_tensor(x1 + update_x1)
        aligned_x2 = self._sanitize_tensor(x2 + update_x2)

        refined_x1, refined_x2 = self.global_refiner(
            aligned_x1=aligned_x1,
            aligned_x2=aligned_x2,
            structure_cue=structure_cue,
        )
        return refined_x1, refined_x2, disagreement_map


class DualFusionDecoder(nn.Module):
    def __init__(
        self,
        high_channels,
        low_channels,
        out_channels,
        img_size,
        cdmoe_balance_weight=0.03,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.cdmoe_fusion = CDMoE(
            img_size=img_size,
            in_channels=low_channels * 2,
            out_channels=low_channels * 2,
            num_experts=4,
            hidden_channels=low_channels,
            noisy_gating=True,
            k=2,
            selector_capacity_weight=0.02,
            expert_balance_weight=cdmoe_balance_weight,
            capacity_exploration_mass=0.08,
        )

        self.conv_low = BasicConv2d(low_channels * 2, out_channels, 1)
        self.conv_high = BasicConv2d(high_channels, out_channels, 1)
        self.conv_cat = BasicConv2d(out_channels * 2, out_channels, 3, padding=1)

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True)
        )

        self.ref_align = nn.Sequential(
            nn.Conv2d(out_channels, low_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels * 2), nn.SiLU(inplace=True)
        )

    def forward(self, high_feature, core_low_feature, whole_low_feature):
        upsampled_high = self.up(high_feature)
        if upsampled_high.shape[-2:] != core_low_feature.shape[-2:]:
            upsampled_high = F.interpolate(
                upsampled_high,
                size=core_low_feature.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        paired_low_feature = torch.cat([core_low_feature, whole_low_feature], dim=1)
        projected_high = self.conv_high(upsampled_high)
        router_reference = self.ref_align(projected_high)

        restored_low_feature, cdmoe_loss, loss_terms = self.cdmoe_fusion(
            x=paired_low_feature,
            ref=router_reference,
        )

        projected_low = self.conv_low(restored_low_feature)
        fused_feature = self.conv_cat(torch.cat([projected_high, projected_low], dim=1))
        return self.final_conv(fused_feature), cdmoe_loss, loss_terms


class DisagreementCalibratedTrustworthyEstimation(nn.Module):
    def __init__(self, feature_channels, num_classes=3, output_scale=4, projection_channels=24):
        super().__init__()
        hidden = max(feature_channels, 32)
        self.num_classes = num_classes

        self.coarse_head = nn.Sequential(
            BasicConv2d(feature_channels, hidden, 3, padding=1),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )

        self.disagreement_gamma = nn.Parameter(torch.ones(1, num_classes, 1, 1))
        self.disagreement_beta = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

        self.dynamic_fusion = nn.Sequential(
            nn.Conv2d(feature_channels + num_classes * 4, hidden, kernel_size=3, padding=1, bias=False),
            _pait_group_norm(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_classes * 4, kernel_size=1),
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(num_classes, feature_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            BasicConv2d(feature_channels, feature_channels, 3, padding=1),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
        )
        self.refine_scale = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        nn.init.zeros_(self.refine[-1].weight)

        self.final_upsample = nn.Upsample(scale_factor=output_scale, mode='bilinear', align_corners=False)
        self.final_feature_head = BasicConv2d(feature_channels, projection_channels, 3, padding=1)
        self.fg_evidence_head = nn.Sequential(
            BasicConv2d(projection_channels, projection_channels, 3, padding=1),
            nn.Conv2d(projection_channels, num_classes * 2, kernel_size=1),
        )
        self.bg_evidence_head = nn.Sequential(
            BasicConv2d(projection_channels, projection_channels, 3, padding=1),
            nn.Conv2d(projection_channels, num_classes * 2, kernel_size=1),
        )
        self.evidence_act = nn.Softplus()
        self.evidence_discount_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.hierarchy_strength = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    @staticmethod
    def _safe_logit(prob, eps=1e-4):
        prob = prob.clamp(eps, 1.0 - eps)
        return torch.log(prob / (1.0 - prob))

    @staticmethod
    def _resize_spatial_tensor(tensor, target_size):
        if tensor.shape[-2:] == target_size:
            return tensor
        leading_shape = tensor.shape[:-2]
        flattened = tensor.reshape(tensor.size(0), -1, tensor.size(-2), tensor.size(-1))
        resized = F.interpolate(flattened, size=target_size, mode='bilinear', align_corners=False)
        return resized.reshape(*leading_shape, *target_size)

    @staticmethod
    def _binary_entropy(prob, eps=1e-6):
        prob = prob.clamp(eps, 1.0 - eps)
        return -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob)) / math.log(2.0)

    def _calibrate_disagreement(self, disagreement_map, target_size):
        if disagreement_map.size(-2) != target_size[0] or disagreement_map.size(-1) != target_size[1]:
            disagreement_map = F.interpolate(disagreement_map, size=target_size, mode='bilinear', align_corners=False)
        disagreement_map = self._sanitize_tensor(disagreement_map, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        disagreement_mean = disagreement_map.mean(dim=(-2, -1), keepdim=True)
        disagreement_std = disagreement_map.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized_disagreement = (disagreement_map - disagreement_mean) / disagreement_std
        return torch.sigmoid(
            self.disagreement_gamma * normalized_disagreement.expand(-1, self.num_classes, -1, -1) +
            self.disagreement_beta
        )

    def _estimate_dce_uncertainty(self, feature, disagreement_map):
        coarse_logits = self.coarse_head(feature)
        coarse_prob = torch.sigmoid(coarse_logits).clamp(1e-6, 1.0 - 1e-6)

        entropy_uncertainty = self._binary_entropy(coarse_prob)
        disagreement = self._calibrate_disagreement(disagreement_map, target_size=feature.shape[-2:])
        coupled_risk = (entropy_uncertainty * disagreement).clamp(0.0, 1.0)
        residual_risk = (entropy_uncertainty - disagreement).abs().clamp(0.0, 1.0)

        fusion_input = torch.cat(
            [feature, entropy_uncertainty, disagreement, coupled_risk, residual_risk],
            dim=1,
        )
        weights = self.dynamic_fusion(fusion_input)
        weights = weights.view(
            weights.size(0), self.num_classes, 4, weights.size(-2), weights.size(-1)
        )
        weights = F.softmax(weights, dim=2)

        uncertainty_cues = torch.stack(
            [entropy_uncertainty, disagreement, coupled_risk, residual_risk],
            dim=2,
        )
        uncertainty_logits = self._safe_logit(uncertainty_cues)
        dce_uncertainty = torch.sigmoid((weights * uncertainty_logits).sum(dim=2)).clamp(0.0, 1.0)

        gate = self.uncertainty_gate(dce_uncertainty)
        scale = 0.20 * torch.sigmoid(self.refine_scale)
        refined_feature = feature + scale * gate * self.refine(feature)
        dcte_meta = {
            "coarse_probability": coarse_prob,
            "coarse_entropy": entropy_uncertainty,
            "calibrated_disagreement": disagreement,
            "coupled_risk": coupled_risk,
            "residual_risk": residual_risk,
            "fusion_weights": weights,
        }
        return self._sanitize_tensor(refined_feature), coarse_logits, dce_uncertainty, dcte_meta

    def _binary_evidence(self, evidence_logits, dce_risk):
        evidence_logits = self._sanitize_tensor(evidence_logits)
        evidence = self.evidence_act(evidence_logits).clamp_min(1e-6)
        b, _, h, w = evidence.shape
        evidence = evidence.view(b, self.num_classes, 2, h, w)
        dce_risk = self._resize_spatial_tensor(dce_risk, target_size=(h, w))
        dce_risk = self._sanitize_tensor(dce_risk, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if dce_risk.size(1) == 1 and self.num_classes > 1:
            dce_risk = dce_risk.expand(-1, self.num_classes, -1, -1)
        if dce_risk.size(1) != self.num_classes:
            raise ValueError(
                f"Expected DCTE risk with 1 or {self.num_classes} channels, got {dce_risk.size(1)}."
            )

        discount_strength = torch.sigmoid(self.evidence_discount_raw)
        discount = (1.0 - discount_strength * dce_risk).clamp(0.0, 1.0)
        evidence = evidence * discount.unsqueeze(2)
        absent_evidence = evidence[:, :, 0]
        present_evidence = evidence[:, :, 1]
        return absent_evidence, present_evidence, discount

    @staticmethod
    def _opinion_from_evidence(absent_evidence, present_evidence, eps=1e-6):
        strength = absent_evidence + present_evidence + 2.0
        belief_absent = absent_evidence / strength.clamp_min(eps)
        belief_present = present_evidence / strength.clamp_min(eps)
        uncertainty = 2.0 / strength.clamp_min(eps)
        return belief_absent, belief_present, uncertainty

    @staticmethod
    def _fuse_binary_opinions(fg_opinion, bg_opinion, eps=1e-4):
        fg_absent, fg_present, fg_uncertainty = fg_opinion
        bg_absent, bg_present, bg_uncertainty = bg_opinion

        conflict = (fg_absent * bg_present + fg_present * bg_absent).clamp(0.0, 1.0 - eps)
        normalizer = (1.0 - conflict).clamp_min(eps)

        belief_absent = (
            fg_absent * bg_absent +
            fg_absent * bg_uncertainty +
            bg_absent * fg_uncertainty
        ) / normalizer
        belief_present = (
            fg_present * bg_present +
            fg_present * bg_uncertainty +
            bg_present * fg_uncertainty
        ) / normalizer
        uncertainty = (fg_uncertainty * bg_uncertainty) / normalizer

        mass_sum = (belief_absent + belief_present + uncertainty).clamp_min(eps)
        belief_present = belief_present / mass_sum
        uncertainty = uncertainty / mass_sum
        probability = (belief_present + 0.5 * uncertainty).clamp(1e-4, 1.0 - 1e-4)
        return probability, uncertainty.clamp(0.0, 1.0)

    def _apply_soft_hierarchy(self, probability, dce_risk, hierarchy_confusion):
        wt_prob = probability[:, 0:1]
        tc_prob = probability[:, 1:2]
        et_prob = probability[:, 2:3]

        contained_tc = torch.maximum(tc_prob, et_prob)
        contained_wt = torch.maximum(wt_prob, contained_tc)
        contained_probability = torch.cat([contained_wt, contained_tc, et_prob], dim=1)
        dce_risk = self._sanitize_tensor(dce_risk, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        hierarchy_confusion = self._sanitize_tensor(
            hierarchy_confusion,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        base_strength = 0.50 * torch.sigmoid(self.hierarchy_strength)
        hierarchy_risk = torch.maximum(dce_risk, hierarchy_confusion)
        hierarchy_risk = hierarchy_risk.amax(dim=1, keepdim=True)
        hierarchy_gate = base_strength * hierarchy_risk
        return (
            (1.0 - hierarchy_gate) * probability +
            hierarchy_gate * contained_probability
        ).clamp(1e-4, 1.0 - 1e-4)

    @staticmethod
    def _hierarchy_confusion_map(probability):
        wt_prob = probability[:, 0:1]
        tc_prob = probability[:, 1:2]
        et_prob = probability[:, 2:3]
        tc_over_wt = F.relu(tc_prob - wt_prob)
        et_over_tc = F.relu(et_prob - tc_prob)
        return torch.cat([
            tc_over_wt,
            torch.maximum(tc_over_wt, et_over_tc),
            et_over_tc,
        ], dim=1).clamp(0.0, 1.0)

    def _trustworthy_estimation(self, feature, coarse_logits, dce_uncertainty):
        feature = self.final_upsample(feature)
        projected_feature = self.final_feature_head(feature)

        coarse_logits_up = F.interpolate(
            coarse_logits,
            size=projected_feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        dce_risk = self._resize_spatial_tensor(
            dce_uncertainty,
            target_size=projected_feature.shape[-2:],
        ).clamp(0.0, 1.0)
        whole_tumor_gate = torch.sigmoid(coarse_logits_up[:, 0:1]).clamp(0.0, 1.0)
        foreground_feature = projected_feature * whole_tumor_gate
        background_feature = projected_feature * (1.0 - whole_tumor_gate)

        foreground_absent, foreground_present, evidence_discount = self._binary_evidence(
            self.fg_evidence_head(foreground_feature),
            dce_risk,
        )
        background_absent, background_present, _ = self._binary_evidence(
            self.bg_evidence_head(background_feature),
            dce_risk,
        )
        foreground_opinion = self._opinion_from_evidence(foreground_absent, foreground_present)
        background_opinion = self._opinion_from_evidence(background_absent, background_present)

        pre_hierarchy_probability, opinion_uncertainty = self._fuse_binary_opinions(
            foreground_opinion,
            background_opinion,
        )
        hierarchy_confusion = self._hierarchy_confusion_map(pre_hierarchy_probability)
        pre_hierarchy_logits = self._safe_logit(pre_hierarchy_probability)
        final_probability = self._apply_soft_hierarchy(
            pre_hierarchy_probability,
            dce_risk,
            hierarchy_confusion,
        )
        final_logits = self._safe_logit(final_probability)
        post_hierarchy_confusion = self._hierarchy_confusion_map(final_probability)
        return {
            "logits": final_logits,
            "coarse_logits": coarse_logits_up,
            "pre_hierarchy_logits": pre_hierarchy_logits,
            "dce_uncertainty": dce_risk,
            "evidence_discount": evidence_discount,
            "opinion_uncertainty": opinion_uncertainty,
            "hierarchy_confusion": hierarchy_confusion,
            "post_hierarchy_confusion": post_hierarchy_confusion,
        }

    def forward(self, feature, disagreement_map):
        feature = self._sanitize_tensor(feature)
        refined_feature, coarse_logits, dce_uncertainty, dcte_meta = self._estimate_dce_uncertainty(
            feature,
            disagreement_map,
        )
        trustworthy_outputs = self._trustworthy_estimation(
            refined_feature,
            coarse_logits,
            dce_uncertainty,
        )
        output_size = trustworthy_outputs["logits"].shape[-2:]
        resized_meta = {
            key: self._resize_spatial_tensor(value, target_size=output_size).clamp(0.0, 1.0)
            for key, value in dcte_meta.items()
        }

        return {
            "logits": self._sanitize_tensor(trustworthy_outputs["logits"]),
            "coarse_logits": self._sanitize_tensor(trustworthy_outputs["coarse_logits"]),
            "pre_hierarchy_logits": self._sanitize_tensor(trustworthy_outputs["pre_hierarchy_logits"]),
            "dce_uncertainty": trustworthy_outputs["dce_uncertainty"].clamp(0.0, 1.0),
            "coarse_probability": resized_meta["coarse_probability"],
            "calibrated_disagreement": resized_meta["calibrated_disagreement"],
            "coarse_entropy": resized_meta["coarse_entropy"],
            "coupled_risk": resized_meta["coupled_risk"],
            "residual_risk": resized_meta["residual_risk"],
            "fusion_weights": resized_meta["fusion_weights"],
            "evidence_discount": trustworthy_outputs["evidence_discount"].clamp(0.0, 1.0),
            "opinion_uncertainty": trustworthy_outputs["opinion_uncertainty"].clamp(0.0, 1.0),
            "hierarchy_confusion": trustworthy_outputs["hierarchy_confusion"].clamp(0.0, 1.0),
            "post_hierarchy_confusion": trustworthy_outputs["post_hierarchy_confusion"].clamp(0.0, 1.0),
        }


class DominantComplementaryModalityFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.core_residual_logits = self._residual_predictor()
        self.whole_residual_logits = self._residual_predictor()
        self.register_buffer(
            "core_prior_logits",
            torch.log(torch.tensor([0.08, 0.42, 0.42, 0.08], dtype=torch.float32)).view(1, 4, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "whole_prior_logits",
            torch.log(torch.tensor([0.42, 0.08, 0.08, 0.42], dtype=torch.float32)).view(1, 4, 1, 1),
            persistent=False,
        )
        self.latest_fusion_stats = {}

    @staticmethod
    def _residual_predictor():
        predictor = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=1, bias=True),
        )
        nn.init.zeros_(predictor[-1].weight)
        nn.init.zeros_(predictor[-1].bias)
        return predictor

    @staticmethod
    def _normalized_entropy(weights, eps=1e-6):
        weights = weights.clamp(eps, 1.0)
        return -(weights * torch.log(weights)).sum(dim=1, keepdim=True) / math.log(weights.size(1))

    @staticmethod
    def _weighted_sum(modality_stack, weights):
        return (modality_stack * weights).sum(dim=1, keepdim=True)

    def forward(self, x):
        if x.dim() != 4 or x.size(1) != 4:
            raise ValueError(f"Expected BraTS input with shape [B, 4, H, W], got {tuple(x.shape)}.")

        flair, t1, t1ce, t2 = x.split(1, dim=1)
        core_weights = F.softmax(self.core_prior_logits + self.core_residual_logits(x), dim=1)
        whole_weights = F.softmax(self.whole_prior_logits + self.whole_residual_logits(x), dim=1)

        core_mixture = self._weighted_sum(x, core_weights)
        whole_mixture = self._weighted_sum(x, whole_weights)
        core_branch = torch.cat([t1ce, t1, core_mixture], dim=1)
        whole_branch = torch.cat([flair, t2, whole_mixture], dim=1)

        with torch.no_grad():
            self.latest_fusion_stats = {
                "core_weight_mean": core_weights.detach().mean(dim=(0, 2, 3)),
                "whole_weight_mean": whole_weights.detach().mean(dim=(0, 2, 3)),
                "core_entropy_mean": self._normalized_entropy(core_weights).detach().mean(),
                "whole_entropy_mean": self._normalized_entropy(whole_weights).detach().mean(),
            }

        return core_branch, whole_branch


class ModalityStructuralPriorBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.et_enhancement_weight_raw = nn.Parameter(self._softplus_inverse(1.0))
        self.et_t1ce_weight_raw = nn.Parameter(self._softplus_inverse(0.5))
        self.tc_t1_weight_raw = nn.Parameter(self._softplus_inverse(0.35))
        self.tc_t1ce_weight_raw = nn.Parameter(self._softplus_inverse(0.25))
        self.tc_fluid_context_weight_raw = nn.Parameter(self._softplus_inverse(0.25))
        self.tc_non_enhancing_core_weight_raw = nn.Parameter(self._softplus_inverse(0.55))
        self.wt_fluid_peak_weight_raw = nn.Parameter(self._softplus_inverse(1.0))
        self.wt_fluid_mean_weight_raw = nn.Parameter(self._softplus_inverse(0.5))
        self.et_bias = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.tc_bias = nn.Parameter(torch.tensor(-1.7, dtype=torch.float32))
        self.wt_bias = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.refiner = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.SiLU(inplace=True),
            nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=False),
        )

    @staticmethod
    def _softplus_inverse(value):
        value = torch.tensor(float(value), dtype=torch.float32)
        return torch.log(torch.expm1(value))

    @staticmethod
    def _positive(raw_value):
        return F.softplus(raw_value)

    def forward(self, x):
        z_flair, z_t1, z_t1ce, z_t2 = x.split(1, dim=1)
        enhance = z_t1ce - z_t1
        non_enhancing_core = F.relu(z_t1 - z_t1ce)
        fluid_mean = 0.5 * (z_flair + z_t2)
        fluid_peak = torch.maximum(z_flair, z_t2)

        et_logit = (
            self._positive(self.et_enhancement_weight_raw) * enhance +
            self._positive(self.et_t1ce_weight_raw) * z_t1ce +
            self.et_bias
        )
        tc_logit = (
            self._positive(self.tc_t1_weight_raw) * z_t1 +
            self._positive(self.tc_t1ce_weight_raw) * z_t1ce +
            self._positive(self.tc_fluid_context_weight_raw) * fluid_mean +
            self._positive(self.tc_non_enhancing_core_weight_raw) * non_enhancing_core +
            self.tc_bias
        )
        wt_logit = (
            self._positive(self.wt_fluid_peak_weight_raw) * fluid_peak +
            self._positive(self.wt_fluid_mean_weight_raw) * fluid_mean +
            self.wt_bias
        )

        prior_logits = torch.cat([wt_logit, tc_logit, et_logit], dim=1)
        wt_logit, tc_logit, et_logit = (prior_logits + 0.1 * self.refiner(prior_logits)).chunk(3, dim=1)

        et_prior = torch.sigmoid(et_logit)
        tc_prior = torch.maximum(torch.sigmoid(tc_logit), et_prior)
        wt_prior = torch.maximum(torch.sigmoid(wt_logit), tc_prior)
        return torch.cat([wt_prior, tc_prior, et_prior], dim=1).clamp(0.0, 1.0)


class CoupledModalityAnatomyPriorBuilder(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_basis=9,
        min_sigma=0.035,
        max_anatomical_strength=0.35,
        interaction_scale=0.10,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_basis = num_basis
        self.min_sigma = float(min_sigma)
        self.max_anatomical_strength = float(max_anatomical_strength)
        self.interaction_scale = float(interaction_scale)

        self.modality_prior = ModalityStructuralPriorBuilder()

        centers, sigmas = self._default_basis_init(num_basis)
        self.mu_raw = nn.Parameter(self._logit(centers.clamp(1e-4, 1.0 - 1e-4)))
        self.sigma_raw = nn.Parameter(self._softplus_inverse((sigmas - self.min_sigma).clamp_min(1e-4)))

        self.class_basis_logits = nn.Parameter(torch.zeros(3, num_basis, dtype=torch.float32))
        self.class_bias = nn.Parameter(torch.tensor([0.15, -0.12, -0.35], dtype=torch.float32).view(1, 3, 1, 1))
        self.anatomical_strength_raw = nn.Parameter(torch.tensor([-1.4, -1.7, -2.0], dtype=torch.float32))

        gate_hidden = 24
        self.coupling_gate = nn.Sequential(
            nn.Conv2d(10, gate_hidden, kernel_size=3, padding=1, bias=False),
            _pait_group_norm(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, gate_hidden, kernel_size=3, padding=1, groups=gate_hidden, bias=False),
            _pait_group_norm(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.coupling_gate[-2].bias, -1.5)

        self.interaction_refiner = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=3, padding=1, bias=False),
            _pait_group_norm(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 3, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.interaction_refiner[-1].weight)
        nn.init.zeros_(self.interaction_refiner[-1].bias)

        self.latest_prior_stats = {}

    @staticmethod
    def _sanitize_tensor(x, posinf=1e4, neginf=-1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)

    @staticmethod
    def _softplus_inverse(value):
        value = torch.as_tensor(value, dtype=torch.float32)
        return torch.log(torch.expm1(value.clamp_min(1e-6)))

    @staticmethod
    def _logit(value):
        value = torch.as_tensor(value, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(value / (1.0 - value))

    @staticmethod
    def _safe_logit(prob, eps=1e-4):
        prob = prob.clamp(eps, 1.0 - eps)
        return torch.log(prob / (1.0 - prob))

    @staticmethod
    def _default_basis_init(num_basis):
        grid_size = int(np.ceil(np.sqrt(num_basis)))
        coords = torch.linspace(0.25, 0.75, grid_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:num_basis]
        if centers.size(0) < num_basis:
            pad = centers.new_full((num_basis - centers.size(0), 2), 0.5)
            centers = torch.cat([centers, pad], dim=0)

        sigmas = torch.full((num_basis, 2), 0.22, dtype=torch.float32)
        if num_basis > 1:
            sigmas[:, 0] = torch.linspace(0.16, 0.26, num_basis)
            sigmas[:, 1] = torch.linspace(0.26, 0.16, num_basis)
        return centers, sigmas

    @torch.no_grad()
    def initialize_anatomical_basis(self, centers, sigmas=None, class_weights=None):
        centers = torch.as_tensor(centers, dtype=self.mu_raw.dtype, device=self.mu_raw.device)
        if centers.dim() != 2 or centers.size(1) != 2:
            raise ValueError(f"centers must have shape [K, 2], got {tuple(centers.shape)}.")
        k = min(centers.size(0), self.num_basis)
        self.mu_raw[:k].copy_(self._logit(centers[:k].clamp(1e-4, 1.0 - 1e-4)))

        if sigmas is not None:
            sigmas = torch.as_tensor(sigmas, dtype=self.sigma_raw.dtype, device=self.sigma_raw.device)
            if sigmas.dim() == 1:
                sigmas = sigmas[:, None].expand(-1, 2)
            if sigmas.dim() != 2 or sigmas.size(1) != 2:
                raise ValueError(f"sigmas must have shape [K] or [K, 2], got {tuple(sigmas.shape)}.")
            self.sigma_raw[:k].copy_(self._softplus_inverse((sigmas[:k] - self.min_sigma).clamp_min(1e-4)))

        if class_weights is not None:
            class_weights = torch.as_tensor(
                class_weights, dtype=self.class_basis_logits.dtype, device=self.class_basis_logits.device
            )
            if class_weights.dim() != 2 or class_weights.size(1) < k or class_weights.size(0) != 3:
                raise ValueError(
                    "class_weights must have shape [3, K] for WT/TC/ET, "
                    f"got {tuple(class_weights.shape)}."
                )
            class_weights = class_weights[:3, :k]
            class_weights = class_weights.clamp_min(1e-6)
            class_weights = class_weights / class_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            self.class_basis_logits[:, :k].copy_(torch.log(class_weights))

    def _normalized_grid(self, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1)

    def build_anatomical_prior(self, x):
        b, _, h, w = x.shape
        grid = self._normalized_grid(h, w, x.device, x.dtype)

        mu = torch.sigmoid(self.mu_raw).to(device=x.device, dtype=x.dtype)
        sigma = (F.softplus(self.sigma_raw).to(device=x.device, dtype=x.dtype) + self.min_sigma).clamp_min(1e-4)

        diff = grid.unsqueeze(0) - mu[:, None, None, :]
        dist = diff[..., 0].square() / sigma[:, 0, None, None].square() + \
               diff[..., 1].square() / sigma[:, 1, None, None].square()
        basis_maps = torch.exp(-0.5 * dist).clamp(0.0, 1.0)
        basis_maps = basis_maps / basis_maps.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

        class_weights = F.softmax(self.class_basis_logits.to(device=x.device, dtype=x.dtype), dim=1)
        anatomical_prior = torch.einsum("ck,khw->chw", class_weights, basis_maps)
        anatomical_prior = anatomical_prior.unsqueeze(0).expand(b, -1, -1, -1)
        anatomical_prior = torch.sigmoid(self._safe_logit(anatomical_prior) + self.class_bias.to(x.device, x.dtype))
        return anatomical_prior.clamp(1e-4, 1.0 - 1e-4)

    def forward(self, x):
        x = self._sanitize_tensor(x)
        modality_prior = self.modality_prior(x).clamp(1e-4, 1.0 - 1e-4)
        anatomical_prior = self.build_anatomical_prior(x)

        gate_input = torch.cat([x, modality_prior, anatomical_prior], dim=1)
        coupling_gate = self.coupling_gate(gate_input)

        anatomical_strength = self.max_anatomical_strength * torch.sigmoid(self.anatomical_strength_raw)
        anatomical_strength = anatomical_strength.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

        interaction = self.interaction_refiner(torch.cat([modality_prior, anatomical_prior], dim=1))
        interaction = torch.tanh(self._sanitize_tensor(interaction))

        logits = self._safe_logit(modality_prior)
        logits = logits + coupling_gate * anatomical_strength * self._safe_logit(anatomical_prior)
        logits = logits + self.interaction_scale * interaction
        unified_prior = torch.sigmoid(self._sanitize_tensor(logits)).clamp(0.0, 1.0)

        pi_wt, pi_tc, pi_et = unified_prior.chunk(3, dim=1)
        pi_et = pi_et.clamp(0.0, 1.0)
        pi_tc = torch.maximum(pi_tc, pi_et)
        pi_wt = torch.maximum(pi_wt, pi_tc)
        unified_prior = torch.cat([pi_wt, pi_tc, pi_et], dim=1).clamp(0.0, 1.0)

        with torch.no_grad():
            self.latest_prior_stats = {
                "modality_prior_mean": modality_prior.mean().detach(),
                "anatomical_prior_mean": anatomical_prior.mean().detach(),
                "coupling_gate_mean": coupling_gate.mean().detach(),
                "anatomical_strength_mean": anatomical_strength.mean().detach(),
                "unified_prior_mean": unified_prior.mean().detach(),
            }

        return unified_prior

class DiTrustNet(nn.Module):
    def __init__(self, num_classes=3, img_size=224, backbone_name="vmamba"):
        super().__init__()
        self.backbone_name = normalize_backbone_name(backbone_name)
        self.modality_fusion = DominantComplementaryModalityFusion()
        self.structural_prior_builder = CoupledModalityAnatomyPriorBuilder(img_size=img_size)

        self.core_backbone = build_backbone(self.backbone_name)
        self.whole_backbone = build_backbone(self.backbone_name)

        stage_channels = get_backbone_dims(self.backbone_name)

        self.alignment_stages = nn.ModuleList([
            PriorGuidedAnisotropicInnovationTransport(dim=channels)
            for channels in stage_channels
        ])

        self.deep_context_gate = GateBlock(stage_channels[3])
        self.mid_context_gate = GateBlock(stage_channels[2])
        self.deep_to_mid_projection = BasicConv2d(stage_channels[3], stage_channels[2], 1)

        self.decoder_stages = nn.ModuleList([
            DualFusionDecoder(
                high_channels=stage_channels[2],
                low_channels=stage_channels[2],
                out_channels=stage_channels[1],
                img_size=img_size // 16,
                cdmoe_balance_weight=0.03,
            ),
            DualFusionDecoder(
                high_channels=stage_channels[1],
                low_channels=stage_channels[1],
                out_channels=stage_channels[0],
                img_size=img_size // 8,
                cdmoe_balance_weight=0.03,
            ),
            DualFusionDecoder(
                high_channels=stage_channels[0],
                low_channels=stage_channels[0],
                out_channels=stage_channels[0] // 2,
                img_size=img_size // 4,
                cdmoe_balance_weight=0.05,
            ),
        ])

        self.auxiliary_heads = nn.ModuleDict({
            "x16": nn.Sequential(
                BasicConv2d(stage_channels[1], 96, 3, padding=1),
                nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
                nn.Conv2d(96, num_classes, 1),
            ),
            "x8": nn.Sequential(
                BasicConv2d(stage_channels[0], 48, 3, padding=1),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                nn.Conv2d(48, num_classes, 1),
            ),
        })
        self.dcte = DisagreementCalibratedTrustworthyEstimation(
            feature_channels=stage_channels[0] // 2,
            num_classes=num_classes,
            output_scale=4,
            projection_channels=24,
        )

    @property
    def backbones(self):
        return self.core_backbone, self.whole_backbone

    def set_cdmoe_routing_schedule(
        self,
        epoch,
        total_epochs,
        warmup_epochs=5,
        temperature_start=2.0,
        temperature_end=0.35,
    ):
        for decoder_stage in self.decoder_stages:
            decoder_stage.cdmoe_fusion.set_routing_schedule(
                epoch=epoch,
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
                temperature_start=temperature_start,
                temperature_end=temperature_end,
            )

    @staticmethod
    def _resize_prior_to_feature_stages(prior_map, feature_stages):
        return [
            F.interpolate(prior_map, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            for feature in feature_stages
        ]

    @staticmethod
    def _mean_resized_maps(source_maps, target_size):
        resized_maps = [
            F.interpolate(source_map, size=target_size, mode='bilinear', align_corners=False)
            for source_map in source_maps
        ]
        return torch.stack(resized_maps, dim=0).mean(dim=0)

    @staticmethod
    def _sum_router_terms(router_terms, term_name):
        terms = [scale_terms[term_name] for scale_terms in router_terms.values()]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    def _align_branch_features(self, core_features, whole_features, structural_prior):
        stage_priors = self._resize_prior_to_feature_stages(structural_prior, core_features)
        aligned_core_features = []
        aligned_whole_features = []
        disagreement_maps = []

        for alignment_stage, core_feature, whole_feature, stage_prior in zip(
            self.alignment_stages,
            core_features,
            whole_features,
            stage_priors,
        ):
            aligned_core, aligned_whole, disagreement_map = alignment_stage(
                core_feature,
                whole_feature,
                stage_prior,
            )
            aligned_core_features.append(aligned_core)
            aligned_whole_features.append(aligned_whole)
            disagreement_maps.append(disagreement_map)

        return aligned_core_features, aligned_whole_features, disagreement_maps

    def load_backbone_weights(self, path):
        return load_pretrained_backbone_pair(
            self.core_backbone,
            self.whole_backbone,
            self.backbone_name,
            path,
        )

    def forward(self, x):
        structural_prior = self.structural_prior_builder(x)
        core_branch, whole_branch = self.modality_fusion(x)

        core_features = self.core_backbone(core_branch)
        whole_features = self.whole_backbone(whole_branch)
        aligned_core_features, aligned_whole_features, disagreement_maps = self._align_branch_features(
            core_features=core_features,
            whole_features=whole_features,
            structural_prior=structural_prior,
        )

        deep_core_context = self.deep_to_mid_projection(
            self.deep_context_gate(aligned_core_features[3])
        )
        deep_whole_context = self.deep_to_mid_projection(
            self.deep_context_gate(aligned_whole_features[3])
        )
        decoder_seed = deep_core_context + deep_whole_context
        mid_core_context = self.mid_context_gate(aligned_core_features[2])
        mid_whole_context = self.mid_context_gate(aligned_whole_features[2])

        decoder_x16, cdmoe_loss_x16, router_terms_x16 = self.decoder_stages[0](
            decoder_seed,
            mid_core_context,
            mid_whole_context,
        )
        aux_logits_x16 = self.auxiliary_heads["x16"](decoder_x16) if self.training else None

        decoder_x8, cdmoe_loss_x8, router_terms_x8 = self.decoder_stages[1](
            decoder_x16,
            aligned_core_features[1],
            aligned_whole_features[1],
        )
        aux_logits_x8 = self.auxiliary_heads["x8"](decoder_x8) if self.training else None

        decoder_x4, cdmoe_loss_x4, router_terms_x4 = self.decoder_stages[2](
            decoder_x8,
            aligned_core_features[0],
            aligned_whole_features[0],
        )

        router_losses = {
            "x16": cdmoe_loss_x16,
            "x8": cdmoe_loss_x8,
            "x4": cdmoe_loss_x4,
        }
        router_terms = {
            "x16": router_terms_x16,
            "x8": router_terms_x8,
            "x4": router_terms_x4,
        }
        total_cdmoe_loss = router_losses["x16"] + router_losses["x8"] + router_losses["x4"]
        cdmoe_sparsity_loss = self._sum_router_terms(router_terms, "sparsity_term")
        cdmoe_capacity_loss = self._sum_router_terms(router_terms, "capacity_term")
        cdmoe_selector_capacity_loss = self._sum_router_terms(
            router_terms,
            "selector_capacity_term",
        )
        cdmoe_balance_loss = self._sum_router_terms(router_terms, "balance_term")

        decoder_disagreement = self._mean_resized_maps(
            source_maps=disagreement_maps,
            target_size=decoder_x4.shape[-2:],
        )
        dcte_outputs = self.dcte(decoder_x4, decoder_disagreement)
        final_logits = dcte_outputs["logits"]
        coarse_logits = dcte_outputs["coarse_logits"]

        meta = {
            "dce_uncertainty": dcte_outputs["dce_uncertainty"],
            "pre_hierarchy_logits": dcte_outputs["pre_hierarchy_logits"],
            "hierarchy_confusion": dcte_outputs["hierarchy_confusion"],
            "post_hierarchy_confusion": dcte_outputs["post_hierarchy_confusion"],
            "coarse_probability": dcte_outputs["coarse_probability"],
            "calibrated_disagreement": dcte_outputs["calibrated_disagreement"],
            "coarse_entropy": dcte_outputs["coarse_entropy"],
            "coupled_risk": dcte_outputs["coupled_risk"],
            "residual_risk": dcte_outputs["residual_risk"],
            "fusion_weights": dcte_outputs["fusion_weights"],
            "evidence_discount": dcte_outputs["evidence_discount"],
            "opinion_uncertainty": dcte_outputs["opinion_uncertainty"],
        }
        return {
            "logits": final_logits,
            "aux_logits": {
                "x16": aux_logits_x16,
                "x8": aux_logits_x8,
                "coarse": coarse_logits,
            },
            "losses": {
                "cdmoe": total_cdmoe_loss,
                "cdmoe_sparsity": cdmoe_sparsity_loss,
                "cdmoe_capacity": cdmoe_capacity_loss,
                "cdmoe_selector_capacity": cdmoe_selector_capacity_loss,
                "cdmoe_balance": cdmoe_balance_loss,
            },
            "meta": meta,
        }

def _format_number(value):
    return f"{value:,}"


def _format_million(value):
    return f"{value / 1e6:.2f} M"


def _print_model_summary(model):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_params = total_params - trainable_params

    print("[Parameters]")
    print(f"Trainable : {_format_number(trainable_params)} ({_format_million(trainable_params)})")
    print(f"Frozen    : {_format_number(frozen_params)} ({_format_million(frozen_params)})")
    print(f"Total     : {_format_number(total_params)} ({_format_million(total_params)})")


def _set_backbone_trainable(model, trainable):
    for backbone in model.backbones:
        for parameter in backbone.parameters():
            parameter.requires_grad = trainable


if __name__ == '__main__':
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiTrustNet(num_classes=3, img_size=img_size).to(device)
    _set_backbone_trainable(model, trainable=False)
    model.eval()

    input_tensor = torch.randn(1, 4, img_size, img_size, device=device)

    print("=============================================")
    print(f"Input: {tuple(input_tensor.shape)}")
    _print_model_summary(model)

    if device.type != "cuda":
        print("[Forward]")
        print("  Skipped: PAIT/SS2D uses CUDA kernels in this project.")
        print("=============================================")
    else:
        with torch.no_grad():
            outputs = model(input_tensor)

        print(f"Output: {tuple(outputs['logits'].shape)}")
        print("=============================================")
