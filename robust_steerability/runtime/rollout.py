"""Reusable HDF5 records for controlled prompt rollouts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


def write_prompt_rollout_group(
    handle: h5py.File | h5py.Group,
    group_name: str,
    record: dict[str, object],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    states: torch.Tensor,
    controls: torch.Tensor,
    next_token_id: int,
    next_token_log_probability: float,
) -> None:
    """Transactionally write one completed prompt rollout."""

    temporary_name = f"tmp_{group_name}"
    if temporary_name in handle:
        del handle[temporary_name]
    group = handle.create_group(temporary_name)
    group.attrs["complete"] = False
    group.attrs["prompt_id"] = str(record["prompt_id"])
    group.attrs["source"] = str(record["source"])
    group.attrs["toxicity"] = float(record["toxicity"])
    group.attrs["text"] = str(record["text"])
    group.attrs["next_token_id"] = next_token_id
    group.attrs["next_token_log_probability"] = next_token_log_probability
    state_array = states.numpy()
    control_array = controls.detach().cpu().float().numpy()
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
        data=input_ids.detach().cpu().numpy().astype(np.int32),
    )
    group.create_dataset(
        "attention_mask",
        data=attention_mask.detach().cpu().numpy().astype(np.uint8),
    )
    group.attrs["complete"] = True
    handle.move(temporary_name, group_name)
    handle.file.flush()


def read_rollout(
    path: Path,
    expected_config_hash: str,
) -> list[dict[str, object]]:
    """Read completed prompt rollouts from one cache file."""

    records = []
    with h5py.File(path, "r") as handle:
        if handle.attrs["config_hash"] != expected_config_hash:
            raise ValueError(f"Incompatible rollout cache: {path}")
        for name in sorted(handle.keys()):
            if name.startswith("tmp_"):
                continue
            group = handle[name]
            records.append(
                {
                    "prompt_id": group.attrs["prompt_id"],
                    "states": torch.from_numpy(group["states"][:]).float(),
                    "controls": torch.from_numpy(group["controls"][:]).float(),
                }
            )
    return records
