"""Small Ours API around unchanged, source-pinned inference definitions."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from checkpoint import load_wm_model
from codec import Mapping, encode_with_mapping
from decision import add_counts, empty
from inversion import InversableStableDiffusionPipeline
from scan import score_maps


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_registry(config):
    root = Path(config['registry'])
    for name, pin in config['registry_sha256'].items():
        if file_sha256(root / name) != pin:
            raise ValueError(f'Registry hash mismatch: {name}')
    positions = np.load(root / 'positions.npy', mmap_mode='r')
    transforms = np.load(root / 'transforms.npy', mmap_mode='r')
    if positions.ndim != 2 or positions.shape[1] != 256 or transforms.shape != positions.shape:
        raise ValueError('Expected aligned N x 256 mapping arrays')
    return positions, transforms


def user_mapping(positions, transforms, user_id):
    if not 0 <= user_id < len(positions):
        raise ValueError('User ID is outside the supplied registry')
    return Mapping(16, 5, list(range(256)), positions[user_id].tolist(), transforms[user_id].tolist())


class Watermarker:
    def __init__(self, config, device='cuda'):
        from diffusers import DPMSolverMultistepScheduler
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if file_sha256(config['checkpoint']) != config['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        self.device = device
        bits = config['message_bits']
        if len(bits) != 64 or set(bits) - {'0', '1'}:
            raise ValueError('Expected exactly 64 binary payload bits')
        self.message = torch.tensor([[int(b) for b in bits]], device=device, dtype=torch.float32)
        self.model, length = load_wm_model(config['checkpoint'], device)
        if length != 64:
            raise ValueError('Expected native 64-bit model')
        self.model.eval().requires_grad_(False)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(config['sd21'], subfolder='scheduler', local_files_only=True)
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            config['sd21'], scheduler=scheduler, torch_dtype=torch.float16,
            revision='fp16', local_files_only=True).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def image(self, path):
        transform = T.Compose([T.ToTensor(), T.Resize((512, 512), antialias=True)])
        with Image.open(path) as image:
            return transform(image.convert('RGB')).unsqueeze(0).to(device=self.device, dtype=torch.float16)

    @torch.inference_mode()
    def embed(self, image, mapping):
        return encode_with_mapping(self.pipe, self.model, image, self.message, mapping)

    @torch.inference_mode()
    def latent(self, image):
        z = self.pipe.get_image_latents(image * 2 - 1, sample=False)
        callbacks = []
        z = self.pipe.forward_diffusion(
            latents=z, text_embeddings=self.pipe.get_text_embedding(''), guidance_scale=1,
            num_inference_steps=15, num_steps=3,
            callback=lambda i, t, v: callbacks.append(int(t)), callback_steps=1)
        if len(callbacks) != 3 or not torch.isfinite(z).all():
            raise ValueError('Native 15/3 inversion did not complete')
        return z

    @torch.inference_mode()
    def extract(self, image, mapping):
        scores = self.model.decoder(mapping.shuffler().shuffle(self.latent(image)).float())
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite decoder output')
        bits = scores > .5
        correct = int((bits == (self.message > .5)).sum())
        return dict(bits=''.join(str(int(b)) for b in bits[0].cpu()), correct_bits=correct,
                    BAR=correct / 64, genuine_TPR51=correct >= 51, genuine_TPR57=correct >= 57,
                    note='Known-mapping detection, not an open-set pool decision')

    @torch.inference_mode()
    def trace(self, image, positions, transforms, pool_sizes):
        pools = sorted(set(pool_sizes))
        if not pools or pools[0] < 2 or pools[-1] > len(positions):
            raise ValueError('Pool sizes must be between 2 and registry size')
        z = self.latent(image)
        state = empty()
        for start in range(0, pools[-1], 100000):
            end = min(start + 100000, pools[-1])
            counts, _ = score_maps(self.model, z, positions[start:end].copy(),
                                   transforms[start:end].copy(), self.message, 256)
            add_counts(state, counts, start, pools)
        if state['n'] != pools[-1]:
            raise ValueError('Incomplete maximum-pool scan')
        return [dict(pool_size=p['pool_size'], max_bits=p['max_bits'], second_bits=p['second_bits'],
                     top1_user_id=p['top1_user_id'], tie_count=p['tie_count'],
                     accepted=bool(p['false_attribution']), reject=bool(p['reject'])) for p in state['pools']]

    @staticmethod
    def save_image(image, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        T.ToPILImage()(image.detach().cpu().float().squeeze(0).clamp(0, 1)).save(path)
