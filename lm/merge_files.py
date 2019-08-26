from pathlib import Path
from typing import *

from fire import Fire
from tqdm import tqdm

import subprocess as sp


def run(*args):
    return sp.check_output(list(map(str, args)))


def wc(filename: Path) -> int:
    return int(run("wc", "-l", filename).split()[0])


def merge_files(save_to: Path, files: Iterable[Path]):
    with Path(save_to).open("w", encoding="utf-8") as fout:
        for fpth in tqdm(files, "Merging files"):
            with Path(fpth).open() as fin:
                for line in tqdm(fin, "Reading lines", wc(fpth)):
                    fout.write(line)


def run_merge_files(save_to: Path, data_root: Path):
    save_to = Path(save_to)
    data_root = Path(data_root)

    files = list(data_root.rglob("*_segmented.txt"))
    merge_files(save_to, files)


def fire_merge_files():
    Fire(run_merge_files)


if __name__ == "__main__":
    fire_merge_files()
