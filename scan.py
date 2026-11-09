"""Unchanged exact candidate scoring, including native batch-one boundary guard."""
import numpy as np
import torch

def patch_bank(z):
    assert z.shape == (1, 4, 64, 64)
    patches = z.reshape(1, 4, 16, 4, 16, 4).permute(0, 2, 4, 1, 3, 5).reshape(256, 4, 4, 4)
    return torch.stack([torch.rot90(patches, k, (-2, -1)) for k in [1, 2, 3]] +
                       [torch.flip(patches, (-1,)), torch.flip(patches, (-2,))])


def map_bank(bank, positions, transforms):
    p = torch.as_tensor(positions, device=bank.device, dtype=torch.long)
    t = torch.as_tensor(transforms, device=bank.device, dtype=torch.long)
    patches = bank[t, torch.arange(256, device=bank.device).expand(len(p), 256)]
    ordered = torch.gather(patches, 1, p[:, :, None, None, None].expand(len(p), 256, 4, 4, 4))
    return ordered.reshape(len(p), 16, 16, 4, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(len(p), 4, 64, 64)


@torch.inference_mode()
def score_maps(model, z, positions, transforms, message, batch, raw=False):
    bank = patch_bank(z)
    counts, logits = [], []
    for i in range(0, len(positions), batch):
        s = model.decoder(map_bank(bank, positions[i:i+batch], transforms[i:i+batch]).float())
        # Resolve rare floating-point boundary cases with the native batch-one call.
        near = (s - .5).abs().amin(-1) <= 1e-5
        for j in torch.nonzero(near, as_tuple=False).flatten().tolist():
            s[j] = model.decoder(map_bank(bank, positions[i+j:i+j+1], transforms[i+j:i+j+1]).float())[0]
        assert torch.isfinite(s).all()
        counts.append(((s > .5) == (message > .5)).sum(-1).cpu().numpy().astype(np.uint8))
        if raw:
            logits.append(s.cpu().numpy())
    return np.concatenate(counts), np.concatenate(logits) if raw else None
