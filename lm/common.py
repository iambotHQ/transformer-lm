import json
from pathlib import Path

from .model import HParams

UNK = "<unk>"
END_OF_LINE = "<endofline>"
END_OF_TEXT = "<endoftext>"


default_hparams = json.loads(
    (Path(__file__).parent / "default_hparams.json").read_text()
)
default_hparams.setdefault("n_hidden", default_hparams["n_embed"])
default_hparams = HParams(**default_hparams)
