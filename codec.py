"""Unchanged native mapping allocator and absolute-encoder embedding."""
import math
import random
from dataclasses import dataclass
from typing import Any
import torch
from mapping import LatentShuffler

@dataclass
class Mapping:
    splits: int
    k: int
    shuffle_indices: list[int]
    position_indices: list[int]
    transform_indices: list[int]

    @property
    def B(self) -> int:
        return self.splits * self.splits

    @property
    def log10_capacity(self) -> float:
        return math.lgamma(self.B + 1) / math.log(10) + self.B * math.log10(self.k)

    def shuffler(self) -> LatentShuffler:
        return LatentShuffler(
            splits=self.splits,
            shuffle_indices=self.shuffle_indices,
            position_indices=self.position_indices,
            transform_indices=self.transform_indices,
        )


def make_mapping(splits: int, k: int, seed: int) -> Mapping:
    if not 1 <= k <= 5:
        raise ValueError("The canonical implementation exposes exactly five local transforms; K must be in [1,5].")
    rng = random.Random(seed)
    B = splits * splits
    return Mapping(
        splits=splits,
        k=k,
        shuffle_indices=rng.sample(range(B), B),
        position_indices=rng.sample(range(B), B),
        transform_indices=[rng.randrange(k) for _ in range(B)],
    )


def mapping_to_json(m: Mapping) -> dict[str, Any]:
    return {"splits": m.splits, "k": m.k, "shuffle_indices": m.shuffle_indices, "position_indices": m.position_indices, "transform_indices": m.transform_indices}


def mapping_from_json(d: dict[str, Any]) -> Mapping:
    return Mapping(int(d["splits"]), int(d["k"]), list(d["shuffle_indices"]), list(d["position_indices"]), list(d["transform_indices"]))


def encode_with_mapping(pipe, wm_model, image: torch.Tensor, message: torch.Tensor, mapping: Mapping) -> torch.Tensor:
    lat = pipe.get_image_latents(image * 2 - 1, sample=False)
    s = mapping.shuffler()
    w = wm_model.encoder(s.shuffle(lat).float(), message, mask=None)
    w = s.unshuffle(w)
    return (pipe.decode_image(w.half()) / 2 + 0.5).clamp(0, 1)
