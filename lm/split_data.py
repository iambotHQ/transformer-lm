import shutil as sh
import subprocess as sp
from pathlib import Path
from typing import *

import numpy as np
import sentencepiece as sntpc
from fire import Fire
from stages import get_logger
from tqdm import tqdm

from .common import END_OF_LINE, END_OF_TEXT

logger = get_logger("Splitter")


def run(*args):
    return sp.check_output(list(map(str, args)))


def wc(filename: Path) -> int:
    return int(run("wc", "-l", filename).split()[0])


def create_split_dirs(data_root: Path) -> Tuple[Path, Path, Path]:
    train_data_path = data_root / "train"
    valid_data_path = data_root / "valid"
    test_data_path = data_root / "test"
    for dir_path in (train_data_path, valid_data_path, test_data_path):
        dir_path.mkdir(exist_ok=True, parents=True)
    return train_data_path, valid_data_path, test_data_path


def split_data2(
    merged_file: Path,
    spmodel: Path,
    out_dir_root: Path = Path("data") / "gpt2" / "encoded",
    train_val_ratio: float = 0.9,
):
    merged_file, out_dir_root, spmodel = list(
        map(Path, [merged_file, out_dir_root, spmodel])
    )
    sp = sntpc.SentencePieceProcessor()
    sp.Load(str(spmodel))

    out_dir_root.mkdir(exist_ok=True, parents=True)
    dtype = np.uint16 if len(sp) < 2 ** 16 - 1 else np.uint32

    eot = sp.PieceToId(END_OF_TEXT)
    eol = sp.PieceToId(END_OF_LINE)

    ids: List[np.ndarray] = []

    all_ids_pth = out_dir_root / "all.npy"

    if all_ids_pth.exists():
        ids = np.load(all_ids_pth)
    else:

        def append_and_clear(x):
            ids.append(np.array(x, dtype=dtype))
            x.clear()

        with merged_file.open(encoding="utf-8") as fin:
            tokens, eots, eols = 0, 0, 0
            pbar_text = (
                lambda tokens, texts, lines: f"Reading texts: {tokens} tokens, {texts} texts, {lines} lines"
            )

            with tqdm(fin, pbar_text(0, 0, 0), wc(merged_file)) as pbar:
                encoded: List[int] = []
                for line in pbar:
                    line = line.strip()

                    if not line:
                        eots += 1
                        tokens += 1
                        encoded.append(eot)
                        append_and_clear(encoded)
                        pbar.set_description(pbar_text(tokens, eots, eols))

                    if len(line) >= 5:
                        tokenized = sp.EncodeAsIds(line)
                        encoded.extend(tokenized)
                        encoded.append(eol)
                        eols += 1
                        tokens += len(tokenized)

                    if len(encoded) > 100000:
                        append_and_clear(encoded)

        ids = np.concatenate(ids)
        np.save(out_dir_root / "all.npy", ids)

    logger.info(f"TOKENS: {ids.shape[0]}")

    # train_tokens_count = int(ids.shape[0] * train_val_ratio)
    train_tokens_count = int(ids.shape[0] - 1e6)

    train_tokens = ids[:train_tokens_count]
    logger.info(f"TRAIN TOKENS: {train_tokens.shape[0]}")

    valid_tokens = ids[train_tokens_count:]
    logger.info(f"VALID TOKENS: {valid_tokens.shape[0]}")

    np.save(out_dir_root / "train.npy", train_tokens)
    np.save(out_dir_root / "valid.npy", valid_tokens)
    np.save(out_dir_root / "test.npy", ids[:10])


def split_data(data_root: Path, out_dir_root: Path, train_val_ratio: float = 0.9):
    data_root = Path(data_root)
    out_dir_root = Path(out_dir_root)
    assert data_root.exists()

    train_data_path, valid_data_path, test_data_path = create_split_dirs(out_dir_root)

    out_dir_root.mkdir(exist_ok=True, parents=True)
    corpus_paths = np.asarray(list(data_root.rglob("*.txt")))
    logger.info("Found", corpus_paths.shape[0], "documents")

    logger.info("Shuffling paths")
    np.random.shuffle(corpus_paths)

    all_count = len(corpus_paths)
    train_count = int(all_count * train_val_ratio)
    valid_count = int(all_count * (1.0 - train_val_ratio))
    test_count = 1  # just because

    logger.info(
        "Data counts:\n\ttrain -",
        train_count,
        "\n\tvalid -",
        valid_count,
        "\n\ttest -",
        test_count,
    )

    filename = lambda idx: f"doc_{idx}.txt"
    for idx, train_path in enumerate(corpus_paths[:train_count]):
        sh.copy(str(train_path), str(train_data_path / filename(idx)))

    for idx, valid_path in enumerate(corpus_paths[train_count:]):
        sh.copy(str(valid_path), str(valid_data_path / filename(idx)))

    for idx, test_path in enumerate(corpus_paths[-test_count:]):
        sh.copy(str(valid_data_path / filename(idx)), test_data_path)


def fire_split_data():
    Fire(split_data2)


if __name__ == "__main__":
    fire_split_data()
