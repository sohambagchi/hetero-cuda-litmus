# TEST-CATALOG.md — Complete Litmus Test Catalog

This document describes every litmus test in the framework: thread operations, memory locations, weak behavior conditions, all valid heterogeneous splits, TB configurations, and scope/variant options.

---

## Table of Contents

1. [2-Thread Tests](#2-thread-tests)
   - [MP — Message Passing](#1-mp--message-passing)
   - [2+2W — Two Plus Two Writes](#2-22w--two-plus-two-writes)
   - [Read-Rel-Sys](#3-read-rel-sys)
   - [Read-Rel-Sys-And-CTA](#4-read-rel-sys-and-cta)
2. [3-Thread Tests](#3-thread-tests)
   - [WRC — Write-Read Causality](#5-wrc--write-read-causality)
   - [RWC — Read-Write Causality](#6-rwc--read-write-causality)
   - [WWC — Write-Write Causality](#7-wwc--write-write-causality)
   - [ISA2 — Instruction Sequence A2](#8-isa2--instruction-sequence-a2)
   - [Z6.1 — Write Serialization](#9-z61--write-serialization)
   - [Z6.3 — Write Serialization Variant](#10-z63--write-serialization-variant)
   - [3.2W — Three Thread Two Writes Each](#11-32w--three-thread-two-writes-each)
   - [WRW+2W — Write-Read-Write Plus Two Writes](#12-wrw2w--write-read-write-plus-two-writes)
3. [4-Thread Tests](#4-thread-tests)
   - [IRIW — Independent Reads of Independent Writes](#13-iriw--independent-reads-of-independent-writes)
   - [IRIW-SC — IRIW with Sequential Consistency Variants](#14-iriw-sc--iriw-with-sequential-consistency-variants)
   - [IRIW-Extended](#15-iriw-extended)
   - [Counterexample — Mixed Scope Test](#16-counterexample--mixed-scope-test)
   - [Paper Example](#17-paper-example)
   - [Paper Example 1](#18-paper-example-1)
   - [Paper Example 2](#19-paper-example-2)
4. [Het Split Enumeration](#het-split-enumeration)
5. [TB Configuration Reference](#tb-configuration-reference)

---

## 2-Thread Tests

### 1. MP — Message Passing

**Category**: Store ordering / message passing

**Memory locations**: 2 (x, y) — uses `2-loc.txt`

**Thread operations**:
```
Thread 0 (producer):         Thread 1 (consumer):
  store x = 1 (relaxed)       r0 = load y (acquire)
  store y = 1 (release)       r1 = load x (relaxed)
```

**Weak behavior**: `r0 == 1 && r1 == 0` — consumer sees the flag (y) but not the data (x).

**Outcome classification**:
| Outcome | Classification |
|---------|---------------|
| r0=1, r1=0 | **weak** |
| All other | other |
| x=0 (test skipped) | na |

**Variants** (from cuda-litmus mp.cu — hardcoded orderings, no `#ifdef` variants):
MP uses fixed release/acquire ordering. For het mode, we add variant support:
- `RELAXED` — all relaxed
- `ACQUIRE` — load y = acquire, load x = relaxed
- `RELEASE` — store y = release, store x = relaxed
- `ACQ_REL` — store y = release, load y = acquire (default)

**TB configurations**: `TB_0_1` (inter-block), `TB_01` (intra-block)

**Het splits**:
| Split | Thread 0 | Thread 1 |
|-------|----------|----------|
| `HET_C0_G1` | CPU | GPU |
| `HET_C1_G0` | GPU | CPU |

**Tuning file** (`tuning-files/mp.txt`):
```
mp 2-loc.txt
HET_C0_G1 HET_C1_G0
TB_0_1 TB_01
SCOPE_SYSTEM SCOPE_DEVICE
RELAXED ACQ_REL
```

---

### 2. 2+2W — Two Plus Two Writes

**Category**: Write coherence cycle

**Memory locations**: 2 (x, y)

**Thread operations**:
```
Thread 0:                    Thread 1:
  store x = 1 (relaxed)       store y = 1 (relaxed)
  FENCE_0()                    FENCE_1()
  store y = 2                  store x = 2
```

**Weak behavior**: `x == 1 && y == 1` — both early stores "won" coherence, meaning the late stores were reordered before the early stores (a coherence cycle).

**Note**: This test checks final memory values, not read results.

**Variants** (from tuning file):
- `RELAXED` — all relaxed, no fences
- `RELEASE` — stores are release
- `BOTH_FENCE` — fences on both threads
- `FENCE_0` — fence on Thread 0 only
- `FENCE_1` — fence on Thread 1 only

**Fence scopes**: `FENCE_SCOPE_DEVICE`, `FENCE_SCOPE_BLOCK`

**TB configurations**: `TB_0_1`, `TB_01`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: `HET_C0_G1`, `HET_C1_G0`

---

### 3. Read-Rel-Sys

**Category**: Scope-specific (device scope release vs system visibility)

**Memory locations**: 2 (x, y)

**Thread operations**:
```
Thread 0:                    Thread 1:
  store x = 1 (relaxed)       store y = 2 (relaxed)
  store y = 1 (release)       r0 = load x (relaxed)
```

**Weak behavior**: `r0 == 0 && y == 2` — Thread 1's y=2 overwrites Thread 0's y=1, but Thread 1 doesn't see x=1.

**Variants**: None (hardcoded orderings). Uses device-scope atomics.

**TB configurations**: `TB_0_1` (always inter-block)

**Het splits**: `HET_C0_G1`, `HET_C1_G0`

**Note**: This test specifically probes whether device-scope release is sufficient for cross-device visibility. In het mode, the interesting split is `HET_C0_G1` where the release store happens on CPU.

---

### 4. Read-Rel-Sys-And-CTA

**Category**: Scope-specific (device vs CTA scope mismatch)

**Memory locations**: 2 (x, y)

**Thread operations**:
```
Thread 0:                          Thread 1:
  store x = 1 (relaxed, device)     store y = 2 (relaxed, device)
  store y = 1 (release, device)     r0 = load x (relaxed, CTA/block)
```

**Weak behavior**: `r0 == 0 && y == 2` — CTA-scoped load on Thread 1 misses device-scoped release from Thread 0.

**Variants**: None (hardcoded with explicit scope casts).

**TB configurations**: `TB_0_1`

**Het splits**: `HET_C0_G1`, `HET_C1_G0`

**Note**: In het mode, the CTA-scope load on Thread 1 is interesting when Thread 1 is on GPU (CTA scope has no meaning on CPU).

---

## 3-Thread Tests

### 5. WRC — Write-Read Causality

**Category**: Multi-copy atomicity / causality

**Memory locations**: 2 (x, y) — uses `THREE_THREAD_TWO_MEM_LOCATIONS()`, `2-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 1                  r0 = load x                  r1 = load y
  (thread_0_store order)       FENCE_1()                    FENCE_2()
                               store y = 1                  r2 = load x
                               (thread_1_store order)       (relaxed)
```

**Weak behavior**: `r0 == 1 && r1 == 1 && r2 == 0` — Thread 1 sees x written, passes message via y to Thread 2, but Thread 2 doesn't see x. Violates multi-copy atomicity.

**Outcome classification** (8 outcomes for 3 reads):
| r0 | r1 | r2 | Classification |
|----|----|----|---------------|
| 1 | 1 | 1 | res0 (seq) |
| 0 | 0 | 0 | res1 (seq) |
| 0 | 0 | 1 | res2 (seq) |
| 0 | 1 | 0 | res3 (seq) |
| 0 | 1 | 1 | res4 (interleaved) |
| 1 | 0 | 0 | res5 (seq) |
| 1 | 0 | 1 | res6 (interleaved) |
| 1 | 1 | 0 | **weak** |

**Variants** (from tuning file):
- Non-fence: `RELAXED`, `ACQUIRE`, `RELEASE`, `ACQ_REL`, `THREAD_0_STORE_RELEASE`
- Fence: `BOTH_FENCE`, `THREAD_1_FENCE`, `THREAD_2_FENCE_ACQ`, `THREAD_2_FENCE_REL`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits** (6 valid splits with at least 1 CPU and 1 GPU):
| Split | T0 | T1 | T2 |
|-------|----|----|-----|
| `HET_C0_G1_G2` | CPU | GPU | GPU |
| `HET_C1_G0_G2` | GPU | CPU | GPU |
| `HET_C2_G0_G1` | GPU | GPU | CPU |
| `HET_C0_C1_G2` | CPU | CPU | GPU |
| `HET_C0_C2_G1` | CPU | GPU | CPU |
| `HET_C1_C2_G0` | GPU | CPU | CPU |

---

### 6. RWC — Read-Write Causality

**Category**: Multi-copy atomicity / causality

**Memory locations**: 2 (x, y) — `2-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 1                  r0 = load x                  store y = 1
  (thread_0_store order)       FENCE_1()                    FENCE_2()
                               r1 = load y                  r2 = load x
```

**Weak behavior**: `r0 == 1 && r1 == 0 && r2 == 0` — Thread 1 sees x but not y; Thread 2 writes y but doesn't see x. Contradicts causality.

**Variants**:
- Non-fence: `RELAXED`, `STORE_SC`, `LOAD_SC`, `THREAD_0_STORE_RELEASE`
- Fence: `BOTH_FENCE`, `THREAD_1_FENCE_STORE_SC`, `THREAD_1_FENCE_LOAD_SC`, `THREAD_2_FENCE`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

### 7. WWC — Write-Write Causality

**Category**: Multi-copy atomicity / write causality

**Memory locations**: 2 (x, y) — `2-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 2                  r0 = load x                  r1 = load y
  (relaxed)                    FENCE_1()                    FENCE_2()
                               store y = 1                  store x = 1
```

**Weak behavior**: `r0 == 2 && r1 == 1 && x == 2` — Thread 1 reads x=2 (from Thread 0), writes y=1, Thread 2 reads y=1, writes x=1, but final x==2 means Thread 0's write is coherence-after Thread 2's write. Creates a non-MCA cycle.

**Variants**:
- Non-fence: `RELAXED`, `ACQ_REL`, `ACQ_ACQ`, `REL_ACQ`, `REL_REL`
- Fence: `BOTH_FENCE`, `THREAD_1_FENCE_ACQ`, `THREAD_1_FENCE_REL`, `THREAD_2_FENCE_ACQ`, `THREAD_2_FENCE_REL`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

### 8. ISA2 — Instruction Sequence A2

**Category**: Multi-copy atomicity / causality chain

**Memory locations**: 3 (x, y, z) — uses `THREE_THREAD_THREE_MEM_LOCATIONS()`, `3-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 1 (relaxed)       r0 = load y                  r1 = load z
  FENCE_0()                    FENCE_1()                    FENCE_2()
  store y = 1                  store z = 1                  r2 = load x
```

**Weak behavior**: `r0 == 1 && r1 == 1 && r2 == 0` — message passes through y→z chain but x not visible to Thread 2.

**Outcome classification** (8 outcomes for 3 reads):
| r0 | r1 | r2 | Classification |
|----|----|----|---------------|
| 1 | 1 | 0 | **weak** |
| (all others) | | | res0..res6, other |

**Variants** (extensive set from tuning file):
- Non-fence: `RELAXED`, `ACQUIRE`, `RELEASE`
- Fence: `ALL_FENCE`, `THREAD_0_FENCE_ACQ`, `THREAD_0_FENCE_REL`, `THREAD_1_FENCE`, `THREAD_2_FENCE_ACQ`, `THREAD_2_FENCE_REL`, `THREAD_0_1_FENCE`, `THREAD_0_2_FENCE_ACQ`, `THREAD_0_2_FENCE_REL`, `THREAD_1_2_FENCE`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

### 9. Z6.1 — Write Serialization

**Category**: Write serialization cycle

**Memory locations**: 3 (x, y, z) — `3-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 2                  store y = 2                  r0 = load z
  store y = 1                  store z = 1                  store x = 1
  (store_order)                (store_order)
```

**Weak behavior**: `x == 2 && y == 2 && r0 == 1` — each thread's second store "wins" coherence on the next location, but Thread 2 reads z=1, forming a 3-way cycle.

**Variants**:
- `RELAXED` — all relaxed
- `ACQ_REL` — release stores

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE` (only)

**Het splits**: Same 6 as WRC.

---

### 10. Z6.3 — Write Serialization Variant

**Category**: Write serialization with fences

**Memory locations**: 3 (x, y, z) — `3-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 1 (relaxed)       store y = 2 (relaxed)        r0 = load z
  FENCE_0()                    FENCE_1()                    FENCE_2()
  store y = 1                  store z = 1                  r1 = load x
```

**Weak behavior**: `y == 2 && r0 == 1 && r1 == 0` — Thread 1's y=2 overwrites Thread 0's y=1, Thread 2 sees z=1 but not x=1.

**Variants**:
- Non-fence: `RELAXED`, `ACQ_REL`
- Fence: `ALL_FENCE`, `FENCE_0`, `FENCE_1`, `FENCE_2`, `FENCE_01`, `FENCE_02`, `FENCE_12`

**Fence scopes**: `FENCE_SCOPE_DEVICE`, `FENCE_SCOPE_BLOCK`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

### 11. 3.2W — Three Thread Two Writes Each

**Category**: 3-way write serialization cycle

**Memory locations**: 3 (x, y, z) — `3-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 2 (relaxed)       store y = 2 (relaxed)        store z = 2 (relaxed)
  FENCE_0()                    FENCE_1()                    FENCE_2()
  store y = 1                  store z = 1                  store x = 1
```

**Weak behavior**: `x == 2 && y == 2 && z == 2` — all three "early" stores (x=2, y=2, z=2) win coherence, meaning each late store (y=1, z=1, x=1) is coherence-ordered before the corresponding early store from the next thread. A 3-way write cycle.

**Note**: Checks final memory values, not reads.

**Variants**:
- Non-fence: `RELAXED`, `RELEASE`
- Fence: `ALL_FENCE`, `FENCE_0`, `FENCE_1`, `FENCE_2`, `FENCE_01`, `FENCE_02`, `FENCE_12`

**Fence scopes**: `FENCE_SCOPE_DEVICE`, `FENCE_SCOPE_BLOCK`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

### 12. WRW+2W — Write-Read-Write Plus Two Writes

**Category**: Mixed write/read causality cycle

**Memory locations**: 2 (x, y) — `2-loc.txt`

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:
  store x = 2                  r0 = load x                  store y = 2 (relaxed)
  (relaxed)                    FENCE_1()                    FENCE_2()
                               store y = 1                  store x = 1
```

**Weak behavior**: `r0 == 2 && x == 2 && y == 2` — Thread 1 reads x=2, Thread 2's x=1 is coherence-before Thread 0's x=2, and Thread 1's y=1 is coherence-before Thread 2's y=2. Non-MCA cycle.

**Variants**:
- Non-fence: `RELAXED`, `ACQUIRE`, `RELEASE`
- Fence: `BOTH_FENCE`, `THREAD_1_FENCE`, `THREAD_2_FENCE_ACQ`, `THREAD_2_FENCE_REL`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations**: `TB_0_1_2`, `TB_01_2`, `TB_0_12`, `TB_02_1`, `TB_012`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits**: Same 6 as WRC.

---

## 4-Thread Tests

### 13. IRIW — Independent Reads of Independent Writes

**Category**: Multi-copy atomicity / independent reads

**Memory locations**: 2 (x, y) — `2-loc.txt`

**Thread operations**:
```
Thread 0:         Thread 1:              Thread 2:         Thread 3:
  store x = 1       r0 = load x            store y = 1       r2 = load y
  (relaxed)         FENCE_1()              (relaxed)         FENCE_3()
                    r1 = load y                              r3 = load x
                    (relaxed)                                (relaxed)
```

**Weak behavior**: `r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0` — observer threads (1 and 3) see writes to x and y in different orders. Classic multi-copy atomicity violation.

**Outcome classification** (comprehensive for 4 reads):
| r0 | r1 | r2 | r3 | Classification |
|----|----|----|-----|---------------|
| 0 | 0 | 0 | 0 | res0 (both observers first) |
| 1 | 1 | 1 | 1 | res1 (both observers last) |
| 0 | 0 | 1 | 1 | res2 |
| 1 | 1 | 0 | 0 | res3 |
| r0==r1, r2≠r3 | | | | res4 (second interleaved) |
| r0≠r1, r2==r3 | | | | res5 (first interleaved) |
| 0 | 1 | 0 | 1 | res6 |
| 0 | 1 | 1 | 0 | res7 |
| 1 | 0 | 0 | 1 | res8 |
| 1 | 0 | 1 | 0 | **weak** |

**Variants**:
- Non-fence: `RELAXED`, `ACQUIRE`
- Fence: `THREAD_1_FENCE`, `THREAD_3_FENCE`, `BOTH_FENCE`

**Fence scopes**: `FENCE_SCOPE_BLOCK`, `FENCE_SCOPE_DEVICE`

**TB configurations** (15 configs for 4 threads):
`TB_0_1_2_3`, `TB_01_2_3`, `TB_01_23`, `TB_0_1_23`, `TB_02_1_3`, `TB_02_13`, `TB_0_2_13`, `TB_03_1_2`, `TB_03_12`, `TB_0_12_3`, `TB_0_123`, `TB_012_3`, `TB_023_1`, `TB_013_2`, `TB_0123`

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Het splits** (14 valid splits for 4-thread test):

1 CPU + 3 GPU (4 splits):
| Split | T0 | T1 | T2 | T3 |
|-------|----|----|----|----|
| `HET_C0_G1_G2_G3` | CPU | GPU | GPU | GPU |
| `HET_C1_G0_G2_G3` | GPU | CPU | GPU | GPU |
| `HET_C2_G0_G1_G3` | GPU | GPU | CPU | GPU |
| `HET_C3_G0_G1_G2` | GPU | GPU | GPU | CPU |

2 CPU + 2 GPU (6 splits):
| Split | T0 | T1 | T2 | T3 |
|-------|----|----|----|----|
| `HET_C0_C1_G2_G3` | CPU | CPU | GPU | GPU |
| `HET_C0_C2_G1_G3` | CPU | GPU | CPU | GPU |
| `HET_C0_C3_G1_G2` | CPU | GPU | GPU | CPU |
| `HET_C1_C2_G0_G3` | GPU | CPU | CPU | GPU |
| `HET_C1_C3_G0_G2` | GPU | CPU | GPU | CPU |
| `HET_C2_C3_G0_G1` | GPU | GPU | CPU | CPU |

3 CPU + 1 GPU (4 splits):
| Split | T0 | T1 | T2 | T3 |
|-------|----|----|----|----|
| `HET_C0_C1_C2_G3` | CPU | CPU | CPU | GPU |
| `HET_C0_C1_C3_G2` | CPU | CPU | GPU | CPU |
| `HET_C0_C2_C3_G1` | CPU | GPU | CPU | CPU |
| `HET_C1_C2_C3_G0` | GPU | CPU | CPU | CPU |

---

### 14. IRIW-SC — IRIW with Sequential Consistency Variants

---

### 15. IRIW-Extended

**Category**: Independent reads / 4-location extension

**Memory locations**: 4

**Purpose**: Repo-specific extension of the IRIW family. This test is present in the current suite and has its own tuning file, so it is part of the supported inventory even though it is not in the original cuda-litmus list.

**TB configurations**: Same 15 as IRIW.

**Scopes**: `SCOPE_DEVICE`, `SCOPE_BLOCK`

**Variants**: `ACQUIRE`, `DEFAULT`

**Het splits**: Same 14 as IRIW.

**Category**: Multi-copy atomicity / SC atomics

**Memory locations**: 2 (x, y)

**Thread operations**: Same as IRIW but with configurable SC annotations.

```
Thread 0:         Thread 1:              Thread 2:         Thread 3:
  store x = 1       r0 = load x            store y = 1       r2 = load y
                    r1 = load y                              r3 = load x
```

**Weak behavior**: `r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0` (same as IRIW)

**Variants** (different from IRIW — tests SC ordering):
- `STORES_SC` — stores are seq_cst
- `FIRST_LOAD_SC` — first load in each observer is seq_cst
- `SECOND_LOAD_SC` — second load in each observer is seq_cst
- `ALL_SC` — all operations seq_cst
- `RELEASE_WRITES` — stores are release
- `ALL_ACQUIRE` — all loads acquire

**TB configurations**: Same 15 as IRIW.

**Het splits**: Same 14 as IRIW.

---

### 16. Counterexample — Mixed Scope Test

**Category**: Scope mismatch demonstration

**Memory locations**: 3 (x, y, z)

**Thread operations** (hardcoded scopes):
```
Thread 0:                    Thread 1:                        Thread 2:                    Thread 3:
  store z = 1                  r0 = load z                      store x = 2                  r1 = load y
  (seq_cst, device)            (seq_cst, CTA/block)             (seq_cst, device)            (acquire, device)
                               store x = 1                      store y = 1                  r2 = load z
                               (seq_cst, CTA/block)             (release, device)            (seq_cst, device)
```

**Weak behavior**: `r0 == 1 && x == 2 && r1 == 1 && r2 == 0` — CTA-scoped operations on Thread 1 don't provide device-level ordering.

**Variants**: None (hardcoded — all scopes are explicit in the test body).

**TB configurations**: 4-thread configs (the interesting configs have T0 and T1 in the same block so CTA scope matters).

**Het splits**: Same 14 as IRIW.

**Note**: This is a critical test — it demonstrates that CTA-scoped SC atomics can break device-level guarantees. In het mode, the most interesting splits put Thread 1 (with CTA scope) on the GPU.

---

### 17. Paper Example

**Category**: Paper-specific 4-thread test

**Memory locations**: 4

**Purpose**: Repo-specific paper example present as `paper-example.cu` and `tuning-files/paper-example.txt`.

**TB configurations**: `TB_0_1_2_3`

**Scopes**: `SCOPE_DEVICE`

**Variants**: `DEFAULT`

**Het splits**: `HET_C0_G1_G2_G3`, `HET_C1_G0_G2_G3`, `HET_C2_G0_G1_G3`, `HET_C3_G0_G1_G2`, `HET_C0_C1_G2_G3`, `HET_C0_C3_G1_G2`, `HET_C2_C3_G0_G1`

---

### 18. Paper Example 1

**Category**: Paper-specific 4-thread test

**Memory locations**: 4 (x, y, z, a)

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:                    Thread 3:
  FENCE_0()                    r0 = load y                  FENCE_2()                    r1 = load a
  store x = 1                  FENCE_1()                    store z = 2                  r2 = load x
  store y = 1                  store z = 1                  store a = 1
```

**Weak behavior**: `r0 == 1 && z == 2 && r1 == 1 && r2 == 0`

**Variants**:
- `DISALLOWED` — acq/rel with SC fences (expected to prevent weak behavior)
- Default — all relaxed, no fences

**TB configurations**: 4-thread configs.

**Het splits**: Same 14 as IRIW.

---

### 19. Paper Example 2

**Category**: Paper-specific 4-thread test (IRIW-like)

**Memory locations**: 4 (x, y, z, a)

**Thread operations**:
```
Thread 0:                    Thread 1:                    Thread 2:                    Thread 3:
  FENCE_0()                    r0 = load y                  FENCE_2()                    r2 = load a
  store x = 1                  FENCE_1()                    store z = 1                  r3 = load x
  store y = 1                  r1 = load z                  store a = 1
```

**Weak behavior**: `r0 == 1 && r1 == 0 && r2 == 1 && r3 == 0` — IRIW-like: observers see writes in different orders.

**Variants**:
- `DISALLOWED` — acq/rel with SC fences
- Default — all relaxed

**TB configurations**: 4-thread configs.

**Het splits**: Same 14 as IRIW.

---

## Het Split Enumeration

### 2-Thread Tests (2 valid splits)

| ID | Split Name | T0 | T1 |
|----|-----------|-----|-----|
| 1 | `HET_C0_G1` | CPU | GPU |
| 2 | `HET_C1_G0` | GPU | CPU |

### 3-Thread Tests (6 valid splits)

| ID | Split Name | T0 | T1 | T2 |
|----|-----------|-----|-----|-----|
| 1 | `HET_C0_G1_G2` | CPU | GPU | GPU |
| 2 | `HET_C1_G0_G2` | GPU | CPU | GPU |
| 3 | `HET_C2_G0_G1` | GPU | GPU | CPU |
| 4 | `HET_C0_C1_G2` | CPU | CPU | GPU |
| 5 | `HET_C0_C2_G1` | CPU | GPU | CPU |
| 6 | `HET_C1_C2_G0` | GPU | CPU | CPU |

### 4-Thread Tests (14 valid splits)

| ID | Split Name | T0 | T1 | T2 | T3 |
|----|-----------|-----|-----|-----|-----|
| 1 | `HET_C0_G1_G2_G3` | CPU | GPU | GPU | GPU |
| 2 | `HET_C1_G0_G2_G3` | GPU | CPU | GPU | GPU |
| 3 | `HET_C2_G0_G1_G3` | GPU | GPU | CPU | GPU |
| 4 | `HET_C3_G0_G1_G2` | GPU | GPU | GPU | CPU |
| 5 | `HET_C0_C1_G2_G3` | CPU | CPU | GPU | GPU |
| 6 | `HET_C0_C2_G1_G3` | CPU | GPU | CPU | GPU |
| 7 | `HET_C0_C3_G1_G2` | CPU | GPU | GPU | CPU |
| 8 | `HET_C1_C2_G0_G3` | GPU | CPU | CPU | GPU |
| 9 | `HET_C1_C3_G0_G2` | GPU | CPU | GPU | CPU |
| 10 | `HET_C2_C3_G0_G1` | GPU | GPU | CPU | CPU |
| 11 | `HET_C0_C1_C2_G3` | CPU | CPU | CPU | GPU |
| 12 | `HET_C0_C1_C3_G2` | CPU | CPU | GPU | CPU |
| 13 | `HET_C0_C2_C3_G1` | CPU | GPU | CPU | CPU |
| 14 | `HET_C1_C2_C3_G0` | GPU | CPU | CPU | CPU |

### Total Combinatorial Space

For a single test, the total number of binary variants is:
```
het_splits × tb_configs × scopes × (non_fence_variants + fence_scopes × fence_variants)
```

Example for WRC: `6 × 5 × 2 × (4 + 2 × 4) = 6 × 5 × 2 × 12 = 720 binaries`

Example for IRIW: `14 × 15 × 2 × (2 + 2 × 3) = 14 × 15 × 2 × 8 = 3,360 binaries`

---

## TB Configuration Reference

### 2-Thread TB Configs (2)

| Config | Description | T0 | T1 |
|--------|------------|----|----|
| `TB_0_1` | Inter-block | Block A | Block B |
| `TB_01` | Intra-block | Block A | Block A |

### 3-Thread TB Configs (5)

| Config | Description | T0 | T1 | T2 |
|--------|------------|----|----|-----|
| `TB_0_1_2` | All inter-block | Block A | Block B | Block C |
| `TB_01_2` | T0,T1 same; T2 different | Block A | Block A | Block B |
| `TB_0_12` | T0 different; T1,T2 same | Block A | Block B | Block B |
| `TB_02_1` | T0,T2 same; T1 different | Block A | Block B | Block A |
| `TB_012` | All intra-block | Block A | Block A | Block A |

### 4-Thread TB Configs (15)

| Config | T0 | T1 | T2 | T3 |
|--------|----|----|----|----|
| `TB_0_1_2_3` | A | B | C | D |
| `TB_01_2_3` | A | A | B | C |
| `TB_01_23` | A | A | B | B |
| `TB_0_1_23` | A | B | C | C |
| `TB_02_1_3` | A | B | A | C |
| `TB_02_13` | A | B | A | B |
| `TB_0_2_13` | A | B | C | B |
| `TB_03_1_2` | A | B | C | A |
| `TB_03_12` | A | B | B | A |
| `TB_0_12_3` | A | B | B | C |
| `TB_0_123` | A | B | B | B |
| `TB_012_3` | A | A | A | B |
| `TB_023_1` | A | B | A | A |
| `TB_013_2` | A | A | B | A |
| `TB_0123` | A | A | A | A |

### TB Config Naming Convention

- Underscores `_` separate groups in **different** thread blocks
- Digits grouped together share the **same** thread block
- Within-block threads use `permute_id` for pairing
- Cross-block threads use `stripe_workgroup` for pairing

### Het Split × TB Config Interaction

For het tests, the TB config only applies to GPU-side threads. If a thread role is on the CPU, its TB position is irrelevant (it's always "external" to all GPU blocks).

When all GPU threads are in the same block (e.g., `TB_012` with `HET_C0_G1_G2`), the GPU threads (T1, T2) share L1 cache. When they're in different blocks, they must communicate through L2/DRAM.

The TB config still matters for het tests because it affects:
1. Which GPU SM the thread is scheduled on
2. Cache hierarchy levels exercised between GPU threads
3. Memory access patterns for multi-GPU-thread instances
