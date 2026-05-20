# Embedded Policy Layer (TinyXAI Scope)

This folder contains the compact policy-layer export path:

1. Generate `beacon_policy.h` from one feature bundle:

```bash
.venv/bin/python scripts/export_policy_to_cpp.py \
  --bundle-dir outputs_composite/part2_extended_v7/tb16_q16_interp \
  --out-header embedded/beacon_policy.h
```

2. Compile/use `embedded/policy_demo.cpp` on host or MCU toolchain.

Notes:
- This is policy-only deployment (`h(a(x))`), not full BEACON audit on MCU.
- Full audit still depends on `(Q+1)` model calls outside this embedded layer.
