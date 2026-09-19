#!/usr/bin/env python3
"""Dump GFPGANv1.4 params_ema grouped into the layout src/gfpgan.cpp expects,
with per-section float-count reconciliation against gfpgan.h's tables.

Usage:
  python dump_layout.py --pth GFPGANv1.4.pth            # verify mapping
  python dump_layout.py --pth GFPGANv1.4.pth --export . # write style.bin
"""
import argparse
import struct
import sys

# (num_output, inc) per style conv — mirrors style_conv_channels in gfpgan.h.
STYLE_CONV_CHANNELS = [(512, 512)] * 8 + [(256, 512), (256, 512), (128, 256),
                                         (128, 128), (64, 128), (64, 64),
                                         (512, 512)]
TO_RGB_CHANNELS = [(3, 512)] * 4 + [(3, 256), (3, 128), (3, 64), (3, 512)]
STYLE_DIM = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--export", metavar="OUT_DIR")
    args = ap.parse_args()

    import torch  # noqa: deferred — this script runs where torch exists.

    state = torch.load(args.pth, map_location="cpu")
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    shapes = {k: tuple(v.shape) for k, v in state.items()}

    def take(prefix):
        got = {k: s for k, s in shapes.items() if k.startswith(prefix)}
        if not got:
            print(f"MISSING: no keys under {prefix}*")
        return got

    def section_floats(out_ch: int, in_ch: int) -> tuple[int, list[str]]:
        """Expected float counts + the state_dict keys for one style conv."""
        keys = [
            f"style_convs.{prefix}.modulated_conv.weight",
            f"style_convs.{prefix}.modulated_conv.modulation.weight",
            f"style_convs.{prefix}.modulated_conv.modulation.bias",
            f"style_convs.{prefix}.conv.weight",
            f"style_convs.{prefix}.conv.bias",
        ]
        counts = [out_ch * in_ch * 9, STYLE_DIM * STYLE_DIM, STYLE_DIM,
                  out_ch * in_ch * 9, out_ch]
        return sum(counts), keys, counts

    ok = True
    blob: list[float] = []
    for i, (out_ch, in_ch) in enumerate(STYLE_CONV_CHANNELS):
        total, keys, counts = section_floats(str(i), in_ch)
        missing = [k for k in keys if k not in shapes]
        shape_ok = all(shapes[k] == tuple(c for c, k2 in zip(counts, keys) if k2 == k)
                       for k in keys if k in shapes)
        status = "ok" if not missing and shape_ok else "MISMATCH"
        if status != "ok":
            ok = False
        print(f"style_convs.{i:<2} out={out_ch:<3} in={in_ch:<3} floats={total:<9} {status}")
        if status == "ok":
            for k in keys:
                blob.extend(state[k].flatten().tolist())

    for i, (out_ch, in_ch) in enumerate(TO_RGB_CHANNELS):
        keys = [
            f"to_rgbs.{i}.modulated_conv.weight",
            f"to_rgbs.{i}.modulated_conv.modulation.weight",
            f"to_rgbs.{i}.modulated_conv.modulation.bias",
            f"to_rgbs.{i}.bias",
        ]
        counts = [out_ch * in_ch, STYLE_DIM * STYLE_DIM, STYLE_DIM, out_ch]
        missing = [k for k in keys if k not in shapes]
        if missing:
            ok = False
            print(f"to_rgbs.{i:<2} MISSING {missing}")
            continue
        print(f"to_rgbs.{i:<2} out={out_ch:<2} in={in_ch:<3} floats={sum(counts):<9} ok")
        for k in keys:
            blob.extend(state[k].flatten().tolist())

    if "stylegan_decoder.constant_input.weight" not in shapes and \
            "constant_input.weight" not in shapes:
        # The constant lives under the decoder module; try both spellings.
        const_key = next((k for k in shapes if k.endswith("constant_input.weight")), None)
    else:
        const_key = "stylegan_decoder.constant_input.weight"
    if const_key:
        n = 4 * 4 * STYLE_DIM
        print(f"const_input      floats={n:<9} found at {const_key} "
              f"{shapes[const_key]}")
        blob.extend(state[const_key].flatten().tolist())
    else:
        ok = False
        print("const_input      MISSING")

    print("MAPPING OK" if ok else "MAPPING INCOMPLETE — see MISSING/MISMATCH above")
    if args.export and ok:
        out = f"{args.export.rstrip('/')}/style.bin"
        with open(out, "wb") as f:
            f.write(struct.pack(f"<{len(blob)}f", *blob))
        print(f"wrote {out} ({len(blob) * 4} bytes)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
