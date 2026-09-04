"""Reusable HDF5 records for generated continuations."""

from __future__ import annotations

import h5py
import numpy as np
import torch


def write_generation_group(
    handle: h5py.File,
    group_name: str,
    record: dict[str, object],
    seed: int,
    prompt_length: int,
    sequences: torch.Tensor,
    generated_ids: torch.Tensor,
    states: torch.Tensor,
    controls: torch.Tensor,
    completion: str,
) -> None:
    """Transactionally write one generated continuation and its trajectory."""

    temporary_name = f"tmp_{group_name}"
    if temporary_name in handle:
        del handle[temporary_name]
    group = handle.create_group(temporary_name)
    group.attrs["complete"] = False
    group.attrs["prompt_id"] = str(record["prompt_id"])
    group.attrs["source"] = str(record["source"])
    group.attrs["prompt_toxicity"] = float(record["toxicity"])
    group.attrs["prompt_text"] = str(record["text"])
    group.attrs["completion"] = completion
    group.attrs["seed"] = seed
    group.attrs["prompt_length"] = prompt_length
    state_array = states.numpy().astype(np.float16)
    control_array = controls.numpy().astype(np.float16)
    group.create_dataset(
        "states",
        data=state_array,
        chunks=(1, min(state_array.shape[1], 16), state_array.shape[2]),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "controls",
        data=control_array,
        chunks=(1, min(control_array.shape[1], 16), control_array.shape[2]),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "input_ids",
        data=sequences.detach().cpu().numpy().astype(np.int32),
    )
    group.create_dataset(
        "generated_ids",
        data=generated_ids.detach().cpu().numpy().astype(np.int32),
    )
    group.attrs["complete"] = True
    handle.move(temporary_name, group_name)
    handle.file.flush()
