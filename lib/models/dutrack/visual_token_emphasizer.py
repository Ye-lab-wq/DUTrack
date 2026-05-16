import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualTokenEmphasizer(nn.Module):
    """TETrack-style learnable visual token keep/reduce predictor."""

    def __init__(self, embed_dim, num_heads=8, hard=False, tau=1.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.hard = hard
        self.tau = tau

        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.head_dim),
                nn.Linear(self.head_dim, self.head_dim),
                nn.GELU(),
            )
            for _ in range(num_heads)
        ])
        self.out_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 2),
                nn.GELU(),
                nn.Linear(self.head_dim // 2, max(self.head_dim // 4, 1)),
                nn.GELU(),
                nn.Linear(max(self.head_dim // 4, 1), 2),
            )
            for _ in range(num_heads)
        ])

    def forward(self, visual_tokens, prev_decision):
        """Return keep logits/probability and cumulative keep decision.

        Args:
            visual_tokens: B x N x C visual tokens.
            prev_decision: B x N x 1 cumulative decision from previous TE layer.
        """
        logits = 0.0
        probs = 0.0
        denom = prev_decision.sum(dim=1, keepdim=True).clamp_min(1.0)

        for head_idx in range(self.num_heads):
            start = head_idx * self.head_dim
            end = start + self.head_dim
            x_head = visual_tokens[:, :, start:end]
            x_head = self.in_proj[head_idx](x_head)

            half_dim = x_head.shape[-1] // 2
            local_x = x_head[:, :, :half_dim]
            global_x = (x_head[:, :, half_dim:] * prev_decision).sum(dim=1, keepdim=True) / denom
            relation = torch.cat([local_x, global_x.expand(-1, x_head.shape[1], -1)], dim=-1)
            head_logits = self.out_proj[head_idx](relation)

            logits = logits + F.log_softmax(head_logits, dim=-1)
            probs = probs + F.softmax(head_logits, dim=-1)

        logits = logits / self.num_heads
        probs = probs / self.num_heads
        tau = max(float(self.tau), 1e-6)
        if self.training:
            keep_decision = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)[:, :, 0:1]
        else:
            keep_decision = probs[:, :, 0:1]
        keep_decision = keep_decision * prev_decision
        return logits, probs, keep_decision
