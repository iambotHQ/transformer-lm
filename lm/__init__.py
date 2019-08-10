import json
from pathlib import Path
from typing import Callable

import torch

from .common import END_OF_LINE, END_OF_TEXT, UNK
from .inference import ModelWrapper, fixed_state_dict
from .model import HParams, Model, OutputGetters, output_getter_type

default_hparams = json.loads(
    (Path(__file__).parent / "default_hparams.json").read_text()
)
default_hparams.setdefault("n_hidden", default_hparams["n_embed"])
default_hparams = HParams(**default_hparams)


def load_model(
    model_path: Path,
    hparams: HParams = default_hparams,
    text_gen_mode: bool = True,
    output_getter: output_getter_type = OutputGetters.mean,
) -> torch.nn.Module:
    model = Model(hparams, text_gen_mode, output_getter)
    state = torch.load(model_path)
    state_dict = fixed_state_dict(state["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()
    return model
