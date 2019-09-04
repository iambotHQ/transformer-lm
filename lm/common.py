import json
from pathlib import Path

from .model import HParams

UNK = "<unk>"
END_OF_LINE = "<endofline>"
END_OF_TEXT = "<endoftext>"


default_hparams = HParams(**{"n_ctx": 1024, "n_embed": 768, "n_head": 12, "n_hidden": 768, "n_layer": 12, "n_vocab": 50000})
