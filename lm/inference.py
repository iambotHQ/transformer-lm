import json
from pathlib import Path
from typing import *

import fire
import numpy as np
import sentencepiece as spm
import torch

from .common import END_OF_LINE, END_OF_TEXT, default_hparams
from .fire_utils import only_allow_defined_args
from .model import HParams, Model, OutputGetters


class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor, device: Union[str, torch.device]):
        self.model = model
        self.sp_model = sp_model
        self.device = device

    @classmethod
    def load(
        cls,
        root: Path,
        device: Union[str, torch.device],
        text_gen_mode: bool = False,
        sp_model: Optional[spm.SentencePieceProcessor] = None,
        encoder_mode: bool = False,
    ):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / "sp.model"))
        hparams = json.loads((root / "params.json").read_text())["hparams"]
        hparams.setdefault("n_hidden", hparams["n_embed"])
        model = Model(HParams(**hparams), text_gen_mode, encoder_mode).to(device)
        state = torch.load(root / "model.pt", map_location=device)
        state_dict = fixed_state_dict(state["state_dict"])
        model.load_state_dict(state_dict)
        return cls(model, sp_model, device)

    @classmethod
    def load_encoder(
        cls, model_path: Path, text_gen_mode: bool, encoder_mode: bool, device: torch.device, output_getter=OutputGetters.mean, params=default_hparams
    ):
        if isinstance(output_getter, str):
            output_getter = getattr(OutputGetters, output_getter)
        model = Model(params, text_gen_mode, encoder_mode, output_getter)
        state_dict = fixed_state_dict(torch.load(model_path, map_location=device)["state_dict"])
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def tokenize(self, s: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(s)

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        assert len(tokens) <= self.model.hparams.n_ctx  # TODO
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(ctx)["logits"].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        return [(float(log_probs[idx, self.token_to_id(token)]), token) for idx, token in enumerate(tokens[1:])]

    def get_next_top_k(self, tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        next_log_probs = self.get_log_probs(tokens)[-1]
        return sorted(((float(next_log_probs[i]), self.id_to_token(i)) for i in next_log_probs.argsort()[-top_k:]), reverse=True)

    def generate_tokens(self, tokens_prefix: List[str], tokens_to_generate: int, top_k: int, no_eot: bool = False, stop_on_eot: bool = False) -> List[str]:
        tokens = ["<endoftext>", *list(tokens_prefix)]
        tok_print = lambda tok: print(tok, end="", flush=True)
        tok_print(f"{self.sp_model.DecodePieces(tokens_prefix)} |")

        ending_puncts = "?!)])>:;}.,"
        starting_puncts = "([{<"

        for _ in range(tokens_to_generate):
            ntk = self.get_next_top_k(tokens, top_k)
            probs = torch.tensor([a[0] for a in ntk]).exp()
            probs /= probs.sum()
            next_token_n = int(probs.multinomial(1))
            next_token = ntk[next_token_n][1]

            del probs
            del next_token_n
            del ntk

            if next_token in [END_OF_TEXT]:
                if no_eot:
                    continue
                if stop_on_eot:
                    break

            tokens.append(next_token)

            normalized_token: str = next_token.replace(END_OF_LINE, "\n").replace(END_OF_TEXT, "\n").replace("▁", " ")
            if (len(normalized_token) > 1 and normalized_token[1] in ending_puncts) or (len(tokens) > 1 and tokens[-2].replace("▁", "") in starting_puncts):
                normalized_token = normalized_token.replace(" ", "")
            tok_print(normalized_token)
        print()

        return tokens

    def generate_tokens_old(self, tokens_prefix: List[str], tokens_to_generate: int, top_k: int) -> List[str]:

        tokens = list(tokens_prefix)

        for i in range(tokens_to_generate):

            # generate TOP_K potential next tokens
            ntk = self.get_next_top_k(tokens, top_k)

            # convert log probs to real probs
            logprobs = np.array(list(map(lambda a: a[0], ntk)))
            probs = np.exp(logprobs) / np.exp(logprobs).sum()

            # pick next token randomly according to probs distribution
            next_token_n = np.random.choice(top_k, p=probs)
            next_token = ntk[next_token_n][1]
            # print (next_token)

            tokens.append(next_token)

        return tokens


def fixed_state_dict(state_dict):
    if all(k.startswith("module.") for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def gen_main(model_path, prefix, tokens_to_generate=42, top_k=8):

    print("loading model from %s" % model_path)
    mw = ModelWrapper.load(Path(model_path))

    print("generating text for prefix %s" % prefix)
    tokens = mw.tokenize(prefix)

    tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k)
    print(mw.sp_model.DecodePieces(tokens_gen))


def fire_gen_main():
    fire.Fire(only_allow_defined_args(gen_main))
