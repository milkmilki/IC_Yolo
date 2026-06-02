from __future__ import annotations

import torch

from train_wm811k_yoloctm import TopologyConditionedCTMBlock, WaferTopologyAdapterGate


def main() -> int:
    torch.manual_seed(7)
    block = TopologyConditionedCTMBlock(
        d_model=96,
        dropout=0.0,
        ring_bins=4,
        sector_bins=8,
        hidden_mult=1.25,
        scale_init=0.02,
    )
    state = torch.zeros(2, 49, 96)
    tokens = torch.randn(2, 49, 96)
    spatial = block(state, tokens, 7, 7)
    fallback = block(state, tokens)
    if spatial.shape != tokens.shape:
        raise AssertionError(f"spatial output shape mismatch: {spatial.shape} != {tokens.shape}")
    if fallback.shape != tokens.shape:
        raise AssertionError(f"fallback output shape mismatch: {fallback.shape} != {tokens.shape}")
    if not torch.isfinite(spatial).all():
        raise AssertionError("spatial output contains non-finite values")
    if not torch.isfinite(fallback).all():
        raise AssertionError("fallback output contains non-finite values")

    adapter_gate = WaferTopologyAdapterGate(d_model=96, ring_bins=4, sector_bins=8, hidden_mult=1.0)
    gate = adapter_gate(tokens, 7, 7)
    if gate.shape != (2, 1, 7, 7):
        raise AssertionError(f"adapter gate shape mismatch: {gate.shape}")
    if not torch.isfinite(gate).all():
        raise AssertionError("adapter gate contains non-finite values")
    print("Topology CTM smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
