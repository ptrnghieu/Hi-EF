"""Hyper-parameters (Appendix A of the paper). """
from dataclasses import dataclass

EMO = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
E2I = {e: i for i, e in enumerate(EMO)}

# set encoder and training
RN = dict(D=128, heads=4, layers=2, dropout=0.2, lr=3e-4, wd=1e-2, epochs=80, patience=12, batch=64,
          aux_w=0.3, a_w=0.3, p_drop_ctx=0.3, p_drop_face=0.15)

# switches of the core model (all on) and of the token-switch model (full model, no ablation)
FULL = dict(slots=True, faces=True, ctx=True, aux=True, mdrop=True)
BASE = dict(readout='query', nq=1, q_readonly=False, slot_emb=True, clip_emb=True, absent='token',
            drop=(), speech_parts=('text', 'audio', 'voice'), clips=(0, 1, 2), seed_offset=0)


@dataclass(frozen=True)
class Constants:
    """Face grouping and frame sampling."""
    pca_dim: int = 128                  # PCA of the HSEmotion embedding (fitted on training faces only)
    maxf: int = 24                      # frames per (slot, clip) cell
    maxf_pool: int = 32                 # frames per clip when all faces are pooled (no-slot variant)
    same_person_cos: float = 0.45       # ArcFace cosine threshold of the average-linkage clustering
    n_face_geo: int = 18                # expression probabilities, valence/arousal, pose, box, mouth, time
    n_voice: int = 9                    # turn descriptor: mouth-audio synchrony per slot, voice similarity, #identities
