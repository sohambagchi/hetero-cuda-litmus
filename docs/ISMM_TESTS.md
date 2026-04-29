# ISMM Tests

This document describes the exact experiment set currently run by `ismm.sh` via `ismm_runner.cu`.

It is intentionally narrower than the main litmus catalog. It documents only the SB and IRIW variants that the ISMM loop actually executes.

## Conventions

- Memory locations: `x`, `y`
- SB weak outcome: `r0 == 0 && r1 == 0`
- IRIW weak outcome: `r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0`
- ARM `rcsc` load: `LDAR`
- ARM `rcpc` load: `LDAPR`
- ARM release store: `STLR`
- PTX "system" and "device" below refer to the scope used by the GPU atomic reference in the runner
- For all IRIW acquire-style reader variants in the current runner, only the first load in each reader is strengthened; the second load remains relaxed
- Each experiment run uses the stress profile from `params-ismm.txt` and `testIterations=100`

## Store Buffering

### Common shape

Thread 0:
```text
store x = 1
r0 = load y
```

Thread 1:
```text
store y = 1
r1 = load x
```

Weak outcome:
```text
r0 == 0 && r1 == 0
```

### 1. `sb-arm-baseline`

Expected: `allowed`

Thread 0 (ARM):
```text
store x = 1        relaxed
r0 = load y       relaxed
```

Thread 1 (ARM):
```text
store y = 1        relaxed
r1 = load x       relaxed
```

### 2. `sb-ptx-baseline`

Expected: `allowed`

Thread 0 (PTX, system scope):
```text
store x = 1        relaxed
r0 = load y       relaxed
```

Thread 1 (PTX, system scope):
```text
store y = 1        relaxed
r1 = load x       relaxed
```

### 3. `sb-arm-rel-acq-rcsc`

Expected: `disallowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
r0 = load y       rcsc acquire (LDAR)
```

Thread 1 (ARM):
```text
store y = 1        release  (STLR)
r1 = load x       rcsc acquire (LDAR)
```

### 4. `sb-arm-rel-acq-rcpc`

Expected: `allowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
r0 = load y       rcpc acquire (LDAPR)
```

Thread 1 (ARM):
```text
store y = 1        release  (STLR)
r1 = load x       rcpc acquire (LDAPR)
```

### 5. `sb-ptx-rel-acq-system`

Expected: `allowed`

Thread 0 (PTX, system scope):
```text
store x = 1        release
r0 = load y       acquire
```

Thread 1 (PTX, system scope):
```text
store y = 1        release
r1 = load x       acquire
```

### 6. `sb-ptx-rel-acq-device`

Expected: `requested`

Thread 0 (PTX, device scope):
```text
store x = 1        release
r0 = load y       acquire
```

Thread 1 (PTX, device scope):
```text
store y = 1        release
r1 = load x       acquire
```

### 7. `sb-mixed-arm-ptx-t0-arm`

Expected: `allowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
r0 = load y       rcsc acquire (LDAR)
```

Thread 1 (PTX, system scope):
```text
store y = 1        release
r1 = load x       acquire
```

### 8. `sb-mixed-arm-ptx-t0-ptx`

Expected: `allowed`

Thread 0 (PTX, system scope):
```text
store x = 1        release
r0 = load y       acquire
```

Thread 1 (ARM):
```text
store y = 1        release  (STLR)
r1 = load x       rcsc acquire (LDAR)
```

## IRIW

### Common shape

Thread 0:
```text
store x = 1
```

Thread 1:
```text
r0 = load x
r1 = load y
```

Thread 2:
```text
store y = 1
```

Thread 3:
```text
r2 = load y
r3 = load x
```

Weak outcome:
```text
r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0
```

### 1. `iriw-arm-baseline`

Expected: `allowed`

Thread 0 (ARM):
```text
store x = 1        relaxed
```

Thread 1 (ARM):
```text
r0 = load x       relaxed
r1 = load y       relaxed
```

Thread 2 (ARM):
```text
store y = 1        relaxed
```

Thread 3 (ARM):
```text
r2 = load y       relaxed
r3 = load x       relaxed
```

### 2. `iriw-ptx-baseline`

Expected: `allowed`

Thread 0 (PTX, system scope):
```text
store x = 1        relaxed
```

Thread 1 (PTX, system scope):
```text
r0 = load x       relaxed
r1 = load y       relaxed
```

Thread 2 (PTX, system scope):
```text
store y = 1        relaxed
```

Thread 3 (PTX, system scope):
```text
r2 = load y       relaxed
r3 = load x       relaxed
```

### 3. `iriw-arm-rel-acq-rcsc`

Expected: `disallowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
```

Thread 1 (ARM):
```text
r0 = load x       rcsc acquire (LDAR)
r1 = load y       relaxed
```

Thread 2 (ARM):
```text
store y = 1        release  (STLR)
```

Thread 3 (ARM):
```text
r2 = load y       rcsc acquire (LDAR)
r3 = load x       relaxed
```

### 4. `iriw-arm-rel-acq-rcpc`

Expected: `disallowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
```

Thread 1 (ARM):
```text
r0 = load x       rcpc acquire (LDAPR)
r1 = load y       relaxed
```

Thread 2 (ARM):
```text
store y = 1        release  (STLR)
```

Thread 3 (ARM):
```text
r2 = load y       rcpc acquire (LDAPR)
r3 = load x       relaxed
```

### 5. `iriw-ptx-rel-acq`

Expected: `allowed`

Thread 0 (PTX, system scope):
```text
store x = 1        release
```

Thread 1 (PTX, system scope):
```text
r0 = load x       acquire
r1 = load y       relaxed
```

Thread 2 (PTX, system scope):
```text
store y = 1        release
```

Thread 3 (PTX, system scope):
```text
r2 = load y       acquire
r3 = load x       relaxed
```

### 6. `iriw-mixed-2rel-ptx-2acq-arm`

Expected: `disallowed`

Thread 0 (PTX, system scope):
```text
store x = 1        release
```

Thread 1 (ARM):
```text
r0 = load x       acquire
r1 = load y       relaxed
```

Thread 2 (PTX, system scope):
```text
store y = 1        release
```

Thread 3 (ARM):
```text
r2 = load y       acquire
r3 = load x       relaxed
```

### 7. `iriw-mixed-2acq-ptx-2rel-arm`

Expected: `allowed`

Thread 0 (ARM):
```text
store x = 1        release  (STLR)
```

Thread 1 (PTX, system scope):
```text
r0 = load x       acquire
r1 = load y       relaxed
```

Thread 2 (ARM):
```text
store y = 1        release  (STLR)
```

Thread 3 (PTX, system scope):
```text
r2 = load y       acquire
r3 = load x       relaxed
```

## Notes

- `ismm.sh` loops over exactly these experiment names in order.
- The runner classifies only the single weak outcome listed above for each test family.
- The `expected` field in `results.csv` is copied directly from the experiment table in `ismm_runner.cu`.
- This document reflects the current implementation, not a broader intended theory space.
