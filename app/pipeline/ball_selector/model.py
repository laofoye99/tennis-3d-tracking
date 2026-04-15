"""V2.2 CandidateTransformer: decoupled selection + existence.

select_head: K classes (which candidate)
exist_head: sigmoid (is ball present)

No more NONE in softmax — selection and existence are independent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 3
NONE_CLASS = K  # kept for memory meta compatibility


class MemoryModule(nn.Module):
    """Memory with split encoding: coords and meta separately."""

    def __init__(self, d_model=64, n_heads=4, k=K):
        super().__init__()
        self.k = k
        self.coords_encoder = nn.Linear(3, d_model)
        self.idx_embed = nn.Embedding(k + 1, d_model)
        self.conf_proj = nn.Linear(1, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, frame_tokens, memory_coords, memory_meta, memory_mask=None):
        B, M = memory_coords.shape[:2]
        coords_embed = self.coords_encoder(memory_coords)
        conf_weights = F.softmax(memory_coords[:, :, :, 2], dim=2)
        coords_pooled = (conf_weights.unsqueeze(-1) * coords_embed).sum(dim=2)

        sel_idx = memory_meta[:, :, 0].long().clamp(0, self.k)
        ball_conf = memory_meta[:, :, 1:2]
        idx_emb = self.idx_embed(sel_idx)
        conf_emb = self.conf_proj(ball_conf)
        mem_tokens = coords_pooled + idx_emb + conf_emb

        if memory_mask is not None and not memory_mask.any():
            return self.norm(frame_tokens)

        attn_mask = None
        if memory_mask is not None:
            no_valid = ~memory_mask.any(dim=1)
            if no_valid.any():
                memory_mask = memory_mask.clone()
                memory_mask[no_valid, 0] = True
            attn_mask = ~memory_mask

        attended, _ = self.cross_attn(
            query=frame_tokens, key=mem_tokens, value=mem_tokens,
            key_padding_mask=attn_mask,
        )
        return self.norm(frame_tokens + attended)


class CandidateTransformer(nn.Module):
    """Decoupled selection + existence.

    select_head: scores over K candidates (which one is the ball)
    exist_head: scalar per frame (is ball present at all)
    """

    def __init__(
        self,
        seq_len=8,
        k=K,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        memory_len=25,
        dropout=0.1,
        candidate_dim=7,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.d_model = d_model
        self.memory_len = memory_len

        self.candidate_embed = nn.Linear(candidate_dim, d_model)
        self.frame_pos_embed = nn.Embedding(seq_len, d_model)
        self.candidate_id_embed = nn.Embedding(k, d_model)
        self.memory = MemoryModule(d_model=d_model, n_heads=n_heads, k=k)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoupled heads
        self.select_proj = nn.Linear(d_model, 1)  # per-candidate score
        self.exist_head = nn.Sequential(          # per-frame existence
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, candidates, memory_coords=None, memory_meta=None, memory_mask=None):
        """
        Args:
            candidates: (B, T, K, 7)
            memory_coords: (B, 25, K, 3) or None
            memory_meta: (B, 25, 2) or None
            memory_mask: (B, 25) bool or None

        Returns:
            select_logits: (B, T, K) — score per candidate
            exist_logits: (B, T) — ball existence logit
        """
        B, T, K_in, D = candidates.shape

        tokens = self.candidate_embed(candidates)
        frame_ids = torch.arange(T, device=tokens.device)
        cand_ids = torch.arange(K_in, device=tokens.device)
        tokens = tokens + self.frame_pos_embed(frame_ids).unsqueeze(0).unsqueeze(2)
        tokens = tokens + self.candidate_id_embed(cand_ids).unsqueeze(0).unsqueeze(1)

        tokens_flat = tokens.reshape(B, T * K_in, self.d_model)

        if memory_coords is not None:
            tokens_flat = self.memory(tokens_flat, memory_coords, memory_meta, memory_mask)

        tokens_flat = self.transformer(tokens_flat)
        tokens_out = tokens_flat.reshape(B, T, K_in, self.d_model)

        # Select: which candidate (softmax over K only)
        select_logits = self.select_proj(tokens_out).squeeze(-1)  # (B, T, K)

        # Exist: ball present? (pool over candidates per frame)
        frame_repr = tokens_out.mean(dim=2)  # (B, T, d_model)
        exist_logits = self.exist_head(frame_repr).squeeze(-1)  # (B, T)

        return select_logits, exist_logits

    def get_predictions(self, select_logits, exist_logits, candidates,
                        exist_threshold=0.5):
        """
        Args:
            select_logits: (B, T, K)
            exist_logits: (B, T)
            candidates: (B, T, K, 7)
            exist_threshold: threshold for existence sigmoid

        Returns:
            selected_idx: (B, T) — 0..K-1 or K=NONE
            coords: (B, T, 2)
            conf: (B, T) — sigmoid(exist_logits)
        """
        # Selection: always pick best candidate
        selected_idx = select_logits.argmax(dim=-1)  # (B, T)

        # Existence: sigmoid threshold
        conf = torch.sigmoid(exist_logits)  # (B, T)
        is_absent = conf < exist_threshold

        # Override to NONE where ball absent
        selected_idx[is_absent] = NONE_CLASS

        # Get coords
        B, T, K_in, _ = candidates.shape
        gather_idx = selected_idx.clamp(0, K_in - 1)
        coords = candidates[:, :, :, :2]
        gather_exp = gather_idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, 2)
        selected_coords = coords.gather(2, gather_exp).squeeze(2)
        selected_coords[is_absent] = 0.0

        return selected_idx, selected_coords, conf

    def build_memory_entry(self, candidates, select_logits, exist_logits):
        """Build memory from current predictions (detached)."""
        with torch.no_grad():
            selected_idx = select_logits.argmax(dim=-1)
            ball_conf = torch.sigmoid(exist_logits)

            mem_coords = candidates[:, :, :, :3].detach()
            mem_meta = torch.stack([
                selected_idx.float(),
                ball_conf,
            ], dim=-1).detach()

        return mem_coords, mem_meta
