"""Training loop of the notebooks (AdamW, early stopping on the UAR of held-out training episodes, gradient clipping)."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import RN
from .metrics import war_uar

HP = {'role': dict(lr=RN['lr'], wd=RN['wd'], epochs=RN['epochs'], patience=RN['patience'], batch=RN['batch'])}
HP['tok'] = HP['role']


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def predict(model, inputs, ix, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            out.append(F.softmax(model(inputs.index(ix[i:i + bs]))[0], -1).cpu())
    return torch.cat(out).numpy()


def train_eval(make_model, hp, inputs, tr, dev, te, seed, device):
    """Fit on rows `tr`, keep the epoch with the best UAR on rows `dev` (patience hp['patience']), return the class
    probabilities of rows `te` and the selection UAR."""
    seed_all(seed)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    y_dev = inputs.yb[dev].cpu().numpy()
    best, best_state, bad = -1, None, 0
    for ep in range(hp['epochs']):
        model.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), hp['batch']):
            j = perm[i:i + hp['batch']]
            logits, aux = model(inputs.index(j), train=True)
            loss = F.cross_entropy(logits, inputs.yb[j])
            for l, t, w in aux.values():
                if (t >= 0).any():
                    loss = loss + w * F.cross_entropy(l, t, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        u = war_uar(predict(model, inputs, dev).argmax(1), y_dev, 7)[1]
        if u > best:
            best, bad = u, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    model.load_state_dict(best_state)
    return predict(model, inputs, te), best
