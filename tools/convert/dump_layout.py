#!/usr/bin/env python3
"""Verify GFPGANv1.4's params_ema against the exact blob layout that
src/gfpgan.cpp:load_weights reads, then optionally export style.bin.

The blob is a raw float32-LE stream consumed strictly sequentially:
  15 × style conv i: [modulated_conv.weight (out·hid·3·3),
                      modulated_conv.modulation.weight (hid·512),
                      modulated_conv.modulation.bias (hid),
                      conv.weight (out·in·k·k, k=3 for i<=7, 1 otherwise),
                      conv.bias (out)]
   8 × to_rgb i:      [modulated_conv.weight (out·hid·1·1),
                       modulated_conv.modulation.weight (hid·512),
                       modulated_conv.modulation.bias (hid),
                       bias (out)]
  const_input:       4·4·512
where (inc, hid, out) come from the C++ tables below. conv.weight sizes of 1
are placeholders — those layers read their skip from the encoder net's blobs,
not from a trained conv.
"""
import argparse
import struct
import sys

# (inc, hid, out) — mirrors style_conv_channels in gfpgan.h.
STYLE_CONV = [(512, 512, 512)] * 8 + [
    (512, 512, 256), (512, 256, 256), (512, 256, 128),
    (512, 128, 128), (512, 128, 64), (512, 64, 64), (512, 512, 512),
]
# (hid, out) — mirrors to_rgb_channels in gfpgan.h (inc is always 512).
TO_RGB = [(512, 3)] * 4 + [(256, 3), (128, 3), (64, 3), (512, 3)]
STYLE_DIM = 512


def style_conv_counts(i: int):
    inc, hid, out = STYLE_CONV[i]
    k = 3 if i <= 7 else 1
    return [(out * hid * 9, ("modulated_conv.weight", (out, hid, 3, 3))),
            (hid * STYLE_DIM, ("modulated_conv.modulation.weight", (hid, STYLE_DIM))),
            (hid, ("modulated_conv.modulation.bias", (hid,))),
            (out * inc * k * k, ("conv.weight", (out, inc, k, k))),
            (out, ("conv.bias", (out,)))]


def to_rgb_counts(i: int):
    hid, out = TO_RGB[i]
    return [(out * hid, ("modulated_conv.weight", (out, hid, 1, 1))),
            (hid * STYLE_DIM, ("modulated_conv.modulation.weight", (hid, STYLE_DIM))),
            (hid, ("modulated_conv.modulation.bias", (hid,))),
            (out, ("bias", (out,)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--export", metavar="OUT_DIR", help="write style.bin (only when all checks pass)")
    args = ap.parse_args()

    import torch

    state = torch.load(args.pth, map_location="cpu")
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]
    shapes = {k: tuple(v.shape) for k, v in state.items()}

    ok = True
    blob: list[float] = []

    for i in range(len(STYLE_CONV)):
        for count, (suffix, expected_shape) in style_conv_counts(i):
            key = f"style_convs.{i}.{suffix}"
            if count == 1:
                continue  # placeholder entry, no trained weights
            shape = shapes.get(key)
            if shape is None:
                print(f"style_convs.{i:<2} {suffix:<45} MISSING")
                ok = False
            elif shape != expected_shape:
                print(f"style_convs.{i:<2} {suffix:<45} SHAPE {shape} != {expected_shape}")
                ok = False
            else:
                blob.extend(state[key].flatten().tolist())
    for i in range(len(TO_RGB)):
        for count, (suffix, expected_shape) in to_rgb_counts(i):
            key = f"to_rgbs.{i}.{suffix}"
            shape = shapes.get(key)
            if shape is None:
                print(f"to_rgbs.{i:<2} {suffix:<45} MISSING")
                ok = False
            elif shape != expected_shape:
                print(f"to_rgbs.{i:<2} {suffix:<45} SHAPE {shape} != {expected_shape}")
                ok = False
            else:
                blob.extend(state[key].flatten().tolist())

    const_key = next((k for k in shapes if k.endswith("constant_input.weight")), None)
    n_const = 4 * 4 * STYLE_DIM
    if const_key and shapes[const_key][0] * shapes[const_key][1] * shapes[const_key][2] == n_const:
        print(f"const_input      {n_const:<9} found at {const_key} {shapes[const_key]}")
        blob.extend(state[const_key].flatten().tolist())
    else:
        print("const_input      MISSING")
        ok = False

    print(f"total blob floats: {len(blob)}")
    print("MAPPING OK" if ok else "MAPPING INCOMPLETE")
    if args.export and ok:
        out = args.export.rstrip("/") + "/style.bin"
        with open(out, "wb") as f:
            f.write(struct.pack(f"<{len(blob)}f", *blob))
        print(f"wrote {out} ({len(blob) * 4} bytes)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
