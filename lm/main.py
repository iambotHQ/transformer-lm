import itertools as it
import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import *

import attr
import fire
import json_log_plots
import numpy as np
import sentencepiece as spm
import torch
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.distributed
import torch.multiprocessing as mp
import torch.utils
import tqdm
from tensorboardX import SummaryWriter
from torch import nn, optim

from .fire_utils import get_defined_args, only_allow_defined_args
from .inference import fixed_state_dict
from .model import HParams, Model


def main(
    run_path,
    dataset_path,
    sp_model_path,
    epochs=10,
    lr=2.5e-4,
    batch_size=2,  # per GPU
    g_accum_gradients=None,  # accumulate gradients N times (globally)
    gradient_checkpointing=False,  # saves GPU memory
    n_ctx=1024,
    n_embed=768,
    n_head=12,
    n_layer=12,
    n_hidden=None,  # equal to n_embed by default (better leave at None)
    clean=False,  # clean run folder
    log_every=1,
    save_every=1000,
    validate_every=None,  # same as save_every by default
    only_validate=False,
    max_tokens=None,
    master_port="40390",
    master_addr="127.0.0.1",
    checkpoint_every=5,  # epochs
    sublog=None,
    # These are set automatically when multiple GPUs are available
    device_id=None,
    n_devices=None,
):
    if n_devices is None:
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            locals_ = locals()
            kwargs = {a: locals_[a] for a in get_defined_args(main)}
            mp.spawn(_main_mp, (kwargs,), n_devices)
            return

    is_main = device_id in {0, None}
    world_size = max(1, n_devices)
    if g_accum_gradients is None:
        g_accum_gradients = world_size
    assert g_accum_gradients % world_size == 0
    accum_gradients = g_accum_gradients // world_size
    if validate_every is None:
        validate_every = save_every

    run_path = Path(run_path)
    model_path = run_path / "model.pt"
    optimizer_path = run_path / "optim.pt"
    if is_main:
        logdir = run_path / "logs"
        if sublog:
            logdir = logdir / sublog
        run_path_mark = run_path / ".lm"
        if clean and run_path.exists():
            assert run_path_mark.exists()  # to avoid removing unrelated folder
            shutil.rmtree(run_path)
        run_path.mkdir(exist_ok=True, parents=True)
        run_path_mark.touch()
        shutil.copy(sp_model_path, run_path / "sp.model")

        log_writer_train = SummaryWriter(logdir / "train", max_queue=5, flush_secs=3)
        log_writer_valid = SummaryWriter(logdir / "valid", max_queue=5, flush_secs=3)

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    hparams = HParams(
        n_vocab=len(sp_model),
        n_ctx=n_ctx,
        n_embed=n_embed,
        n_hidden=n_hidden or n_embed,
        n_head=n_head,
        n_layer=n_layer,
        gradient_checkpointing=gradient_checkpointing,
    )

    params = dict(
        hparams=attr.asdict(hparams),
        argv=" ".join(sys.argv),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        g_accum_gradients=g_accum_gradients,
    )

    params_s = json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False)
    if is_main:
        print(params_s)
        (run_path / "params.json").write_text(params_s, encoding="utf8")

    dataset_path = Path(dataset_path)
    print(f"Loading dataset from {dataset_path}")
    valid_dataset = np.load(dataset_path / "valid.npy")
    train_dataset = np.load(dataset_path / "train.npy")
    step_tokens = n_ctx * batch_size * g_accum_gradients  # all GPUs
    print(f"Train dataset has {len(train_dataset):,} tokens")
    print(f"Validation dataset has {len(valid_dataset):,} tokens")

    if torch.cuda.is_available():
        device = torch.device("cuda", index=device_id)
    else:
        device = torch.device("cpu")

    model = Model(hparams, True, False).to(device)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_meter = AverageMeter()
    cudnn.benchmark = True

    seen_tokens = 0

    def load_model():
        """ Load model, update seen_tokens value
        """
        nonlocal seen_tokens
        state = torch.load(model_path, map_location=device)
        if "seen_tokens" in state:
            seen_tokens = state["seen_tokens"]
        else:  # legacy format
            seen_tokens = state["step"] * step_tokens
        state_dict = fixed_state_dict(state["state_dict"])
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        print(f"Resuming from seen_tokens {seen_tokens:,}")

    if model_path.exists():
        load_model()

    if device_id is not None:
        print(f"device {device} initializing process group")
        os.environ["MASTER_PORT"] = master_port
        os.environ["MASTER_ADDR"] = master_addr
        torch.distributed.init_process_group(backend="nccl", rank=device_id, world_size=world_size)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        print(f"process group for {device} initialized")

    def loss_fn(logits, ctx):
        return cross_entropy(input=logits[:, :-1].reshape([-1, logits.shape[-1]]), target=ctx[:, 1:].reshape(-1))

    def train_step(context: torch.LongTensor):
        """ Train step on one GPU.
        """
        if context is None:
            context = _gen_training_batch(train_dataset, n_ctx=n_ctx, batch_size=batch_size * accum_gradients)
            context = torch.LongTensor(context)

        optimizer.zero_grad()
        loss_scale = n_ctx * batch_size * accum_gradients / (512 * 4 * 32)
        for ctx in torch.split(context, batch_size):
            ctx = ctx.to(device=device)
            logits = model(ctx)["logits"]
            loss = loss_fn(logits, ctx)
            (loss * loss_scale).backward()
            loss_meter.update(float(loss.item()))

        optimizer.step()

    def train():
        nonlocal seen_tokens
        epoch_size = len(train_dataset) // step_tokens * step_tokens
        pbar = tqdm.trange(epochs, desc="epochs", dynamic_ncols=True, disable=not is_main)
        init_epoch_pbar = lambda: tqdm.trange(epoch_size, dynamic_ncols=True, disable=not is_main)
        epoch_pbar = init_epoch_pbar()
        pbar.update(seen_tokens // epoch_size)
        pbar.refresh()
        epoch_pbar.update(seen_tokens % epoch_size)
        step = 0
        epoch, train_loss = 0, 0.0
        # context_gen = _gen_training_batch(train_dataset, n_ctx=n_ctx, batch_size=batch_size * accum_gradients)
        context = None

        avg_epoch_loss, avg_epoch_perplexity = [], []
        while seen_tokens < epochs * epoch_size:
            if max_tokens and seen_tokens >= max_tokens:
                print(f"max_tokens {max_tokens} reached, " f"saving and exiting")
                save()
                validate(epoch)
                return

            # context = torch.LongTensor(next(context_gen)) # TODO GSBATCH
            train_step(context)

            seen_tokens += step_tokens
            step += 1

            epoch_pbar.update(step_tokens)
            epoch_pbar.set_description(f"epoch {1 + epoch}")
            epoch_pbar.set_postfix(loss=f"{loss_meter.mean():.2f}")
            epoch_pbar.refresh()

            if step % save_every == 0:
                save()

            if (epoch + 1) % checkpoint_every == 0:
                save(f"model-{epoch}epochs.pt")

            if is_main and step % log_every == 0:
                train_loss = loss_meter.mean()
                json_log_plots.write_event(run_path, step=seen_tokens, loss=train_loss)
                loss_meter.reset()

                avg_epoch_loss.append(train_loss)
                avg_epoch_perplexity.append(np.exp(train_loss))

                log_writer_train.add_scalar("loss_iter", float(train_loss), seen_tokens)
                log_writer_train.add_scalar("perplexity_iter", float(np.exp(train_loss)), seen_tokens)

            if step % validate_every == 0:
                validate(epoch)

            if seen_tokens % epoch_size == 0:
                pbar.update()
                epoch_pbar.close()
                epoch_pbar = init_epoch_pbar()

                if is_main:
                    log_writer_train.add_scalar("loss_epoch", sum(avg_epoch_loss) / len(avg_epoch_loss), epoch)
                    log_writer_train.add_scalar(
                        "perplexity_epoch", sum(avg_epoch_perplexity) / len(avg_epoch_perplexity), epoch
                    )
                    avg_epoch_loss.clear()
                    avg_epoch_perplexity.clear()

                epoch += 1

        # end of training
        save()
        validate(epoch)

    def validate(epoch: int):
        if not is_main or world_size != 1:
            return
        valid_loss = get_valid_loss()
        json_log_plots.write_event(run_path, step=seen_tokens, valid_loss=valid_loss)
        log_writer_valid.add_scalar("loss_epoch", valid_loss, epoch)
        log_writer_valid.add_scalar("perplexity_epoch", np.exp(valid_loss), epoch)

    def get_valid_loss():
        """ Run validation, return mean loss. This is a pessimistic score,
        as validation contexts are non-overlapping.
        """
        model.eval()
        losses = AverageMeter()
        with torch.no_grad():
            for ctx in _valid_batch_iter(valid_dataset, batch_size=batch_size, n_ctx=n_ctx):
                if not ctx:
                    continue
                ctx = torch.LongTensor(ctx).to(device)
                logits = model(ctx)["logits"]
                loss = loss_fn(logits, ctx)
                losses.update(float(loss.item()))
        model.train()
        return losses.mean()

    def save(name=None):
        if not is_main:
            return
        for path in [model_path, optimizer_path]:
            if path.exists():
                shutil.copy(path, run_path / f"{path.stem}-prev{path.suffix}")
        model_to_save = {"state_dict": _unwrapped_model(model).state_dict(), "seen_tokens": seen_tokens}

        if name:
            torch.save(model_to_save, run_path / name)

        torch.save(model_to_save, model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

    if only_validate:
        if world_size != 1:
            print("multi-GPU validation is not supported yet")
            sys.exit(1)
        if is_main:
            print(f"Validation loss: {get_valid_loss():.4f}")
    else:
        try:
            train()
        except KeyboardInterrupt:
            if is_main:
                log_writer_train.close()
                log_writer_valid.close()
                print("Interrupted, saving")
                save()
                sys.exit(1)


def batch_ids_generator(data: np.ndarray, batch_size: int) -> Generator[List[int], None, None]:
    gen = iter(it.cycle(data))
    while gen:
        yield [next(gen) for _ in range(batch_size)]


def _gen_training_batch(dataset: np.ndarray, n_ctx: int, batch_size: int):
    # for batch_ids in batch_ids_generator(dataset, batch_size):
    #     yield torch.LongTensor([dataset[idx:idx + n_ctx] for idx in batch_ids])
    indices = [np.random.randint(0, len(dataset) - n_ctx) for _ in range(batch_size)]
    return [dataset[idx : idx + n_ctx] for idx in indices]


def _valid_batch_iter(dataset: np.ndarray, *, batch_size: int, n_ctx: int):
    start_indices = range(0, len(dataset) - n_ctx, n_ctx)
    return _batch_it(
        (
            dataset[start_idx : start_idx + n_ctx]
            for start_idx in tqdm.tqdm(start_indices, desc="validation", leave=False)
        ),
        batch_size=batch_size,
    )


def _batch_it(it, batch_size: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _unwrapped_model(model: nn.Module) -> nn.Module:
    """ Return underlying model without data paraller wrapper.
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class AverageMeter:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def mean(self):
        return statistics.mean(self.values)

    def reset(self):
        self.values.clear()


def _main_mp(i, kwargs):
    """ Wrapper to use with mp.spawn.
    """
    kwargs["device_id"] = i
    return main(**kwargs)


def fire_main():
    fire.Fire(only_allow_defined_args(main))
