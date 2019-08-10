import json
from pathlib import Path

from .common import END_OF_LINE, END_OF_TEXT, UNK
from .inference import ModelWrapper
from .model import HParams, Model

default_hparams = json.loads((Path(__file__).parent / "default_params.json").read_text())["hparams"]
default_hparams.setdefault("n_hidden", default_hparams["n_embed"])
