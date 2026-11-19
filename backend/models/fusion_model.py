import torch
import torch.nn as nn
import numpy as np

class ModalityEncoder(nn.Module):
    """
    Projects 14-D unimodal vectors [probs (7), logits (7)] into shared embedding space.
    """
    def __init__(self, in_features: int = 14, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class LearnedAttentionFusionNet(nn.Module):
    """
    Learned Cross-Modal Multi-Head Attention Fusion Network with Dynamic Shannon-Entropy Reliability Gating.
    Supports dynamic missing-modality key_padding_mask for unimodal, bimodal, and trimodal inputs.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.audio_enc = ModalityEncoder(in_features=14, embed_dim=embed_dim)
        self.text_enc = ModalityEncoder(in_features=14, embed_dim=embed_dim)
        self.video_enc = ModalityEncoder(in_features=14, embed_dim=embed_dim)

        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor, video_feat: torch.Tensor, mask: torch.Tensor = None):
        b = audio_feat.shape[0]
        device = audio_feat.device

        if mask is None:
            mask = torch.ones((b, 3), dtype=torch.bool, device=device)

        h_a = self.audio_enc(audio_feat)
        h_t = self.text_enc(text_feat)
        h_v = self.video_enc(video_feat)

        tokens = torch.stack([h_a, h_t, h_v], dim=1)

        key_padding_mask = ~mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False

        attn_out, _ = self.mha(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm1(tokens + attn_out)

        # Dynamic Shannon Entropy computation for reliability gating
        p_a = torch.clamp(audio_feat[:, :7], min=1e-8)
        p_t = torch.clamp(text_feat[:, :7], min=1e-8)
        p_v = torch.clamp(video_feat[:, :7], min=1e-8)

        log7 = float(np.log(7.0))
        ent_a = (-(p_a * torch.log(p_a)).sum(dim=-1, keepdim=True) / log7).unsqueeze(1)
        ent_t = (-(p_t * torch.log(p_t)).sum(dim=-1, keepdim=True) / log7).unsqueeze(1)
        ent_v = (-(p_v * torch.log(p_v)).sum(dim=-1, keepdim=True) / log7).unsqueeze(1)
        entropies = torch.cat([ent_a, ent_t, ent_v], dim=1)

        gate_input = torch.cat([tokens, 1.0 - entropies], dim=-1)
        raw_gates = self.gate_fc(gate_input)

        mask_expanded = mask.unsqueeze(-1).float()
        effective_gates = raw_gates * mask_expanded
        gate_sums = effective_gates.sum(dim=1, keepdim=True).clamp(min=1e-7)
        normalized_gates = effective_gates / gate_sums

        fused_repr = (tokens * normalized_gates).sum(dim=1)
        logits = self.classifier(fused_repr)

        return logits, normalized_gates.squeeze(-1)

class AdvancedCrossModalAttentionFusionNet(nn.Module):
    """
    Advanced Cross-Modal Multi-Head Transformer Fusion Network with Consensus Residuals
    and Uncertainty-Aware Reliability Gates.
    Input per modality: 16-D vector [probs (7), centered_logits (7), certainty (1), quality (1)].
    """
    def __init__(self, in_features: int = 16, embed_dim: int = 96, num_heads: int = 4, num_layers: int = 2, num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Modality Encoders
        self.audio_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Modality Type Embedding (Audio=0, Text=1, Video=2)
        self.modality_type_emb = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        # Multi-Head Self/Cross Attention Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Dynamic Reliability Gate MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 2, 48),
            nn.GELU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

        # Direct Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Learned residual gating parameter
        self.consensus_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, audio_feat: torch.Tensor, text_feat: torch.Tensor, video_feat: torch.Tensor, mask: torch.Tensor = None):
        b = audio_feat.shape[0]
        device = audio_feat.device

        if mask is None:
            mask = torch.ones((b, 3), dtype=torch.bool, device=device)

        # 1. Project modalities to shared latent space
        h_a = self.audio_proj(audio_feat)
        h_t = self.text_proj(text_feat)
        h_v = self.video_proj(video_feat)

        tokens = torch.stack([h_a, h_t, h_v], dim=1) + self.modality_type_emb

        # 2. Transformer Cross-Attention with masking
        key_padding_mask = ~mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False

        attended = self.transformer_encoder(tokens, src_key_padding_mask=key_padding_mask)
        attended = self.norm(attended)

        # 3. Dynamic Reliability Gates
        # Extract certainty and quality indicators from features: indices 14 and 15
        cert_qual_a = audio_feat[:, 14:16].unsqueeze(1)
        cert_qual_t = text_feat[:, 14:16].unsqueeze(1)
        cert_qual_v = video_feat[:, 14:16].unsqueeze(1)
        cq = torch.cat([cert_qual_a, cert_qual_t, cert_qual_v], dim=1)

        gate_in = torch.cat([attended, cq], dim=-1)
        raw_gates = self.gate_mlp(gate_in)

        mask_expanded = mask.unsqueeze(-1).float()
        effective_gates = raw_gates * mask_expanded
        gate_sums = effective_gates.sum(dim=1, keepdim=True).clamp(min=1e-7)
        normalized_gates = effective_gates / gate_sums

        # 4. Neural Representation
        fused_repr = (attended * normalized_gates).sum(dim=1)
        neural_logits = self.classifier(fused_repr)

        # 5. Consensus Residual Head: centered log-odds from indices 7:14
        cl_a = audio_feat[:, 7:14].unsqueeze(1)
        cl_t = text_feat[:, 7:14].unsqueeze(1)
        cl_v = video_feat[:, 7:14].unsqueeze(1)
        cls_all = torch.cat([cl_a, cl_t, cl_v], dim=1)
        consensus_logits = (cls_all * normalized_gates).sum(dim=1)

        # Total Fused Logits
        gamma = torch.clamp(self.consensus_weight, 0.1, 0.9)
        final_logits = (1.0 - gamma) * neural_logits + gamma * consensus_logits

        return final_logits, normalized_gates.squeeze(-1)


