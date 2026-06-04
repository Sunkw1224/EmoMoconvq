"""
emo_cross_trans.py
==================
EmoMoconvq -- emotion-conditioned extension of MoConVQ's T2M-MoConGPT.

This file is the *new contribution* of the EmoMoconvq project. It does NOT
modify any original MoConVQ source file. Instead it subclasses the pretrained
temporal transformer and injects a discrete emotion-conditioning signal, so
the baseline (`cross_trans_ori_fixsum.Text2Motion_Transformer`) stays intact
and runnable for ablation.

Baseline (frozen):
    MoConVQCore.Model.cross_trans_ori_fixsum.Text2Motion_Transformer

Extension (this file):
    * EmoCrossCondTransFeature   -- temporal transformer + emotion fusion
    * EmoText2Motion_Transformer -- drop-in emotion-conditioned GPT

Emotion fusion (proposal Eq. 2):
        f_cond = f_T5 + MLP_proj(E_e(c))
where f_T5 is the (header-processed) T5 text feature consumed by the temporal
transformer's cross-attention layers, c is a discrete emotion label, E_e is a
trainable emotion embedding table and MLP_proj is a two-layer MLP.

Note on dimensions: the T5-large feature consumed by cross-attention is
1024-d (`bert_dim`), so MLP_proj maps  emo_dim -> 1024  (the proposal's text
that "d_e matches the temporal-transformer hidden dim" is imprecise; the
fusion happens in the 1024-d text-feature space).

Graceful-degradation init: the LAST layer of MLP_proj is zero-initialised, so
at the start of fine-tuning  MLP_proj(E_e(c)) == 0  and  f_cond == f_T5 : the
model is numerically identical to the pretrained baseline.
"""

import torch
import torch.nn as nn

from MoConVQCore.Model.cross_trans_ori_fixsum import (
    Text2Motion_Transformer,
    CrossCondTransFeature,
)

# Discrete emotion vocabulary  C  (proposal Sec. III-B, |C| = 5)
EMOTION_TO_IDX = {
    "neutral": 0,
    "happy":   1,
    "sad":     2,
    "angry":   3,
    "fearful": 4,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}
NUM_EMOTIONS = len(EMOTION_TO_IDX)


class EmoCrossCondTransFeature(CrossCondTransFeature):
    """Temporal transformer with an additive emotion-conditioning branch.

    Identical to the pretrained ``CrossCondTransFeature`` plus two new modules:
      * ``emotion_embedding`` -- E_e, trainable table (num_emotions x emo_dim)
      * ``emotion_proj``      -- MLP_proj, projects e_c into the T5 space

    The emotion label for the current forward pass is held in the transient
    attribute ``current_emotion`` (set by ``EmoText2Motion_Transformer``).
    """

    def __init__(self, num_vq=1024, embed_dim=512, clip_dim=512, block_size=16,
                 num_layers=2, n_head=8, drop_out_rate=0.1, fc_rate=4,
                 bert_dim=1024, bert_max_len=512,
                 num_emotions=NUM_EMOTIONS, emo_dim=512):
        super().__init__(num_vq, embed_dim, clip_dim, block_size, num_layers,
                         n_head, drop_out_rate, fc_rate, bert_dim, bert_max_len)
        self.num_emotions = num_emotions
        self.emo_dim = emo_dim
        self.bert_dim = bert_dim

        # --- EmoMoconvq contribution: emotion-conditioning modules ----------
        # E_e : discrete emotion id -> trainable embedding vector
        self.emotion_embedding = nn.Embedding(num_emotions, emo_dim)
        # MLP_proj : emo_dim -> bert_dim, two-layer MLP (proposal Sec. III-B)
        self.emotion_proj = nn.Sequential(
            nn.Linear(emo_dim, emo_dim),
            nn.GELU(),
            nn.Linear(emo_dim, bert_dim),
        )

        # --- initialisation -------------------------------------------------
        # NOTE: these modules are created AFTER super().__init__ has already
        # run self.apply(self._init_weights), so the init below is preserved.
        nn.init.normal_(self.emotion_embedding.weight, mean=0.0, std=0.02)
        # zero-init the LAST projection layer -> delta == 0 at step 0,
        # so f_cond == f_T5 and the model starts identical to the baseline.
        nn.init.zeros_(self.emotion_proj[-1].weight)
        nn.init.zeros_(self.emotion_proj[-1].bias)

        # transient: emotion ids for the current forward pass (None = baseline)
        self.current_emotion = None

    def emotion_delta(self, emotion, ref):
        """Return MLP_proj(E_e(c)) shaped (B, 1, bert_dim), or None.

        ``ref`` is any tensor used to infer the target device.
        """
        if emotion is None:
            return None
        if not torch.is_tensor(emotion):
            emotion = torch.as_tensor(emotion)
        emotion = emotion.long().to(ref.device).reshape(-1)
        e_c = self.emotion_embedding(emotion)           # (B, emo_dim)
        delta = self.emotion_proj(e_c).unsqueeze(1)     # (B, 1, bert_dim)
        return delta

    def forward(self, latents, clip_feature, bert_feature, bert_mask):
        # ---- begin: identical to CrossCondTransFeature.forward ------------
        if len(latents) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            token_embeddings = torch.cat(
                [self.cond_emb(clip_feature).unsqueeze(1), latents], dim=1)
        x = self.pos_embed(token_embeddings)
        bert_feature = self.bert_header(bert_feature, src_key_padding_mask=bert_mask)
        # ---- end: identical part -----------------------------------------

        # ==== EmoMoconvq contribution: additive emotion fusion ============
        #      f_cond = f_T5 + MLP_proj(E_e(c))            (proposal Eq. 2)
        delta = self.emotion_delta(self.current_emotion, bert_feature)
        if delta is not None:
            bert_feature = bert_feature + delta   # broadcast over text tokens
        # ==================================================================

        for blk in self.blocks:
            x = blk(x, bert_feature, bert_mask)
        return x


class EmoText2Motion_Transformer(Text2Motion_Transformer):
    """Drop-in emotion-conditioned replacement for ``Text2Motion_Transformer``.

    Usage::

        gpt = EmoText2Motion_Transformer(**vars(gpt_config()), embeddings=emb)
        gpt.load_state_dict(pretrained_state_dict, strict=False)   # emo params kept
        gpt.configure_finetuning(num_temporal_layers_to_tune=4)
        logits, _ = gpt(latents, idxs, clip_feature, bert_feat, bert_mask,
                        emotion=emotion_ids)

    Passing ``emotion=None`` makes it behave *exactly* like the baseline, which
    is convenient for ablation and keeps the original scripts working.
    """

    def __init__(self, num_emotions=NUM_EMOTIONS, emo_dim=512, **kwargs):
        super().__init__(**kwargs)
        # Replace the temporal transformer with the emotion-aware variant.
        # NT = 12 layers, matching the value hardcoded in the original
        # Text2Motion_Transformer.__init__.
        self.trans_temporal = EmoCrossCondTransFeature(
            num_vq=kwargs.get("num_vq", 1024),
            embed_dim=kwargs.get("embed_dim", 512),
            clip_dim=kwargs.get("clip_dim", 512),
            block_size=kwargs.get("block_size", 16),
            num_layers=12,
            n_head=kwargs.get("n_head", 8),
            drop_out_rate=kwargs.get("drop_out_rate", 0.1),
            fc_rate=kwargs.get("fc_rate", 4),
            num_emotions=num_emotions,
            emo_dim=emo_dim,
        )

    # -- thread the emotion label through forward / sample -----------------
    def forward(self, latents, idxs, clip_feature, bert_feature, bert_mask,
                emotion=None):
        self.trans_temporal.current_emotion = emotion
        try:
            return super().forward(latents, idxs, clip_feature,
                                   bert_feature, bert_mask)
        finally:
            self.trans_temporal.current_emotion = None

    def sample(self, clip_feature, bert_feature, bert_mask, emotion=None,
               **kwargs):
        self.trans_temporal.current_emotion = emotion
        try:
            return super().sample(clip_feature, bert_feature, bert_mask,
                                  **kwargs)
        finally:
            self.trans_temporal.current_emotion = None

    # -- parameter-efficient fine-tuning setup ----------------------------
    def configure_finetuning(self, num_temporal_layers_to_tune=4, verbose=True):
        """Freeze the pretrained model; unfreeze only the EmoMoconvq parameters
        and the top-K layers of the temporal transformer (proposal Sec. III-C).

        Trainable after this call:
          * emotion_embedding (E_e)
          * emotion_proj      (MLP_proj)
          * the top-K CrossBlock layers of the temporal transformer
        Everything else (encoder, codebook, physics decoder, depth transformer,
        bert_header, lower temporal layers) stays frozen.
        """
        for p in self.parameters():
            p.requires_grad = False

        emo = self.trans_temporal
        for p in emo.emotion_embedding.parameters():
            p.requires_grad = True
        for p in emo.emotion_proj.parameters():
            p.requires_grad = True

        n_blocks = len(emo.blocks)
        K = max(0, min(num_temporal_layers_to_tune, n_blocks))
        for i in range(n_blocks - K, n_blocks):
            for p in emo.blocks[i].parameters():
                p.requires_grad = True

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_emo = (sum(p.numel() for p in emo.emotion_embedding.parameters())
                 + sum(p.numel() for p in emo.emotion_proj.parameters()))
        if verbose:
            print("[EmoMoconvq] fine-tuning configuration")
            print(f"  temporal transformer layers : {n_blocks} "
                  f"(tuning top {K})")
            print(f"  emotion module params       : {n_emo:,}")
            print(f"  trainable params            : {n_train:,} "
                  f"({100.0 * n_train / n_total:.2f}% of {n_total:,})")
        return {"total": n_total, "trainable": n_train,
                "emotion": n_emo, "tuned_layers": K, "num_blocks": n_blocks}
