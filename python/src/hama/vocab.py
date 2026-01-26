from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, List


@dataclass
class Vocabulary:
    encoder: List[str]
    decoder: List[str]

    @classmethod
    def load(cls, path: Path | None = None) -> "Vocabulary":
        if path is None:
            with resources.files("hama.assets").joinpath("g2p_vocab.json").open("r", encoding="utf-8") as rf:
                data = json.load(rf)
        else:
            with open(path, "r", encoding="utf-8") as rf:
                data = json.load(rf)
        return cls(encoder=data["encoder"], decoder=data["decoder"])

    @property
    def encoder_token_to_id(self) -> Dict[str, int]:
        return {token: idx for idx, token in enumerate(self.encoder)}

    @property
    def decoder_token_to_id(self) -> Dict[str, int]:
        return {token: idx for idx, token in enumerate(self.decoder)}

    @property
    def pad_id(self) -> int:
        return self.decoder_token_to_id["<pad>"]

    @property
    def sos_id(self) -> int:
        return self.decoder_token_to_id["<sos>"]

    @property
    def eos_id(self) -> int:
        return self.decoder_token_to_id["<eos>"]
