#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpSpec:
    kind: str
    location: str
    order: str
    result_slot: int | None = None


@dataclass(frozen=True)
class ThreadSpec:
    domain: str
    scope: str
    ops: tuple[OpSpec, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    expectation: str
    test_kind: str
    threads: tuple[ThreadSpec, ...]


@dataclass(frozen=True)
class HelperAsm:
    cpu_store_relaxed: tuple[str, ...]
    cpu_store_release: tuple[str, ...]
    cpu_load_relaxed: tuple[str, ...]
    cpu_load_acquire: tuple[str, ...]
    cpu_load_rcsc: tuple[str, ...]
    cpu_load_rcpc: tuple[str, ...]
    gpu_load_relaxed: tuple[str, ...]
    gpu_load_acquire_system: tuple[str, ...]
    gpu_load_acquire_device: tuple[str, ...]
    gpu_store_relaxed: tuple[str, ...]
    gpu_store_release_system: tuple[str, ...]
    gpu_store_release_device: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Verify ISMM runner codegen and per-variant operation order."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=root,
        help="Repository root",
    )
    parser.add_argument(
        "--nvcc",
        default="/usr/local/cuda-12.4/bin/nvcc",
        help="Path to nvcc",
    )
    parser.add_argument(
        "--objdump",
        default="objdump",
        help="Path to objdump",
    )
    parser.add_argument(
        "--cuobjdump",
        default="cuobjdump",
        help="Path to cuobjdump",
    )
    parser.add_argument(
        "--host-gpu-arch",
        default="sm_90",
        help="GPU arch used for the primary host+device build",
    )
    parser.add_argument(
        "--cuobjdump-gpu-arch",
        default="sm_87",
        help="GPU arch used for the cuobjdump helper build",
    )
    parser.add_argument(
        "--host-arch-flags",
        default="-march=armv8.3-a+rcpc",
        help="Host compiler flags passed through nvcc",
    )
    parser.add_argument(
        "--mem-def",
        default="MEM_MALLOC",
        help="Memory backend define used for verification builds",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def run_command(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        cmd = " ".join(command)
        fail(f"command failed: {cmd}\n{result.stdout}{result.stderr}")
    return result.stdout


def ensure_in_order(text: str, snippets: list[str], label: str) -> None:
    position = -1
    for snippet in snippets:
        index = text.find(snippet, position + 1)
        if index < 0:
            fail(f"missing ordered snippet in {label}: {snippet}")
        position = index


def normalize_doc_order(order_text: str) -> str:
    text = order_text.strip().lower()
    if "rcsc acquire" in text:
        return "rcsc"
    if "rcpc acquire" in text:
        return "rcpc"
    if "release" in text:
        return "release"
    if "acquire" in text:
        return "acquire"
    if "relaxed" in text:
        return "relaxed"
    fail(f"unknown order text: {order_text}")


def parse_doc_ops(block_lines: list[str]) -> tuple[OpSpec, ...]:
    ops: list[OpSpec] = []
    for line in block_lines:
        stripped = line.strip()
        if not stripped:
            continue
        store_match = re.match(r"store ([xy]) = 1\s+(.*)", stripped)
        if store_match:
            ops.append(
                OpSpec(
                    kind="store",
                    location=store_match.group(1),
                    order=normalize_doc_order(store_match.group(2)),
                )
            )
            continue
        load_match = re.match(r"r(\d+) = load ([xy])\s+(.*)", stripped)
        if load_match:
            ops.append(
                OpSpec(
                    kind="load",
                    location=load_match.group(2),
                    order=normalize_doc_order(load_match.group(3)),
                    result_slot=int(load_match.group(1)),
                )
            )
            continue
        fail(f"could not parse doc op line: {stripped}")
    return tuple(ops)


def parse_doc_experiments(doc_text: str) -> dict[str, ExperimentSpec]:
    experiments: dict[str, ExperimentSpec] = {}
    current_test_kind = ""
    lines = doc_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("## Store Buffering"):
            current_test_kind = "sb"
        elif line.startswith("## IRIW"):
            current_test_kind = "iriw"

        experiment_match = re.match(r"### \d+\. `([^`]+)`", line)
        if not experiment_match:
            index += 1
            continue

        name = experiment_match.group(1)
        section_lines: list[str] = []
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if next_line.startswith("## ") or re.match(r"### \d+\. `([^`]+)`", next_line):
                break
            section_lines.append(next_line)
            index += 1

        expectation_match = re.search(r"Expected: `([^`]+)`", "\n".join(section_lines))
        if not expectation_match:
            fail(f"missing expectation for {name}")

        thread_specs: dict[int, ThreadSpec] = {}
        line_index = 0
        while line_index < len(section_lines):
            thread_match = re.match(r"Thread (\d+) \((ARM|PTX)(?:, (system|device) scope)?\):", section_lines[line_index])
            if not thread_match:
                line_index += 1
                continue

            thread_index = int(thread_match.group(1))
            domain = "cpu" if thread_match.group(2) == "ARM" else "gpu"
            scope = "system" if domain == "cpu" else thread_match.group(3)
            if scope is None:
                fail(f"missing GPU scope for {name} thread {thread_index}")

            if line_index + 1 >= len(section_lines) or section_lines[line_index + 1].strip() != "```text":
                fail(f"missing code block for {name} thread {thread_index}")

            block_index = line_index + 2
            block_lines: list[str] = []
            while block_index < len(section_lines) and section_lines[block_index].strip() != "```":
                block_lines.append(section_lines[block_index])
                block_index += 1
            if block_index >= len(section_lines):
                fail(f"unterminated code block for {name} thread {thread_index}")

            thread_specs[thread_index] = ThreadSpec(
                domain=domain,
                scope=scope,
                ops=parse_doc_ops(block_lines),
            )
            line_index = block_index + 1

        if not thread_specs:
            fail(f"no threads parsed for {name}")

        ordered_threads = tuple(thread_specs[i] for i in range(max(thread_specs) + 1))
        experiments[name] = ExperimentSpec(
            name=name,
            expectation=expectation_match.group(1),
            test_kind=current_test_kind,
            threads=ordered_threads,
        )

    if not experiments:
        fail("no experiments parsed from docs")
    return experiments


def extract_braced_block(text: str, start_index: int) -> tuple[str, int]:
    if text[start_index] != "{":
        fail("brace extractor did not start on '{'")
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1], index + 1
    fail("unterminated brace block")


def parse_source_op(match: re.Match[str]) -> OpSpec:
    kind = match.group(1)
    args = [arg.strip() for arg in match.group(2).split(",")]
    if kind == "Store":
        if len(args) != 2:
            fail(f"bad Store args: {match.group(0)}")
        return OpSpec(
            kind="store",
            location=args[0].replace("LOC_", "").lower(),
            order=args[1].replace("MEM_ORDER_", "").lower(),
        )
    if len(args) != 3:
        fail(f"bad Load args: {match.group(0)}")
    return OpSpec(
        kind="load",
        location=args[0].replace("LOC_", "").lower(),
        order=args[1].replace("MEM_ORDER_", "").lower(),
        result_slot=int(args[2]),
    )


def parse_source_thread(line: str) -> ThreadSpec:
    if line == "kNoThread":
        return ThreadSpec(domain="none", scope="none", ops=())
    if line.startswith("CpuThread("):
        domain = "cpu"
        scope = "system"
    elif line.startswith("GpuSystemThread("):
        domain = "gpu"
        scope = "system"
    elif line.startswith("GpuDeviceThread("):
        domain = "gpu"
        scope = "device"
    else:
        fail(f"unknown thread constructor: {line}")

    ops = tuple(parse_source_op(match) for match in re.finditer(r"(Store|Load)\(([^)]*)\)", line))
    return ThreadSpec(domain=domain, scope=scope, ops=ops)


def parse_source_experiments(source_text: str) -> dict[str, ExperimentSpec]:
    marker = "static constexpr ExperimentConfig kExperiments[] = {"
    start = source_text.find(marker)
    if start < 0:
        fail("could not find kExperiments in source")
    array_start = source_text.find("{", start)
    array_block, _ = extract_braced_block(source_text, array_start)
    inner = array_block[1:-1]

    experiment_blocks: list[str] = []
    depth = 0
    block_start: int | None = None
    for index, char in enumerate(inner):
        if char == "{":
            if depth == 0:
                block_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and block_start is not None:
                experiment_blocks.append(inner[block_start:index + 1])
                block_start = None

    experiments: dict[str, ExperimentSpec] = {}
    for block in experiment_blocks:
        strings = re.findall(r'"([^"]+)"', block)
        if len(strings) < 2:
            fail(f"could not parse experiment strings from block:\n{block}")
        name, expectation = strings[0], strings[1]
        kind_match = re.search(r"\b(TEST_SB|TEST_IRIW)\b", block)
        count_match = re.search(r"\bTEST_[A-Z]+,\s*(\d+),", block)
        if not kind_match or not count_match:
            fail(f"could not parse kind/count for {name}")
        test_kind = "sb" if kind_match.group(1) == "TEST_SB" else "iriw"
        thread_count = int(count_match.group(1))

        parsed_threads: list[ThreadSpec] = []
        for line in block.splitlines():
            stripped = line.strip().rstrip(",")
            if stripped.startswith(("CpuThread(", "GpuSystemThread(", "GpuDeviceThread(", "kNoThread")):
                parsed_threads.append(parse_source_thread(stripped))

        if len(parsed_threads) < thread_count:
            fail(f"not enough threads parsed for {name}")

        experiments[name] = ExperimentSpec(
            name=name,
            expectation=expectation,
            test_kind=test_kind,
            threads=tuple(parsed_threads[:thread_count]),
        )

    if not experiments:
        fail("no experiments parsed from source")
    return experiments


def compare_experiments(doc_specs: dict[str, ExperimentSpec], source_specs: dict[str, ExperimentSpec]) -> None:
    if set(doc_specs) != set(source_specs):
        missing = sorted(set(doc_specs) - set(source_specs))
        extra = sorted(set(source_specs) - set(doc_specs))
        fail(f"experiment set mismatch. missing={missing} extra={extra}")

    for name in sorted(doc_specs):
        doc_spec = doc_specs[name]
        source_spec = source_specs[name]
        if doc_spec != source_spec:
            fail(
                "experiment mismatch for "
                f"{name}\n"
                f"doc:    {doc_spec}\n"
                f"source: {source_spec}"
            )


def extract_function_body(source_text: str, signature_prefix: str) -> str:
    start = source_text.find(signature_prefix)
    if start < 0:
        fail(f"could not find function signature: {signature_prefix}")
    brace_index = source_text.find("{", start)
    block, _ = extract_braced_block(source_text, brace_index)
    return block


def verify_generic_execution_order(source_text: str) -> None:
    cpu_body = extract_function_body(source_text, "void execute_cpu_thread(")
    ensure_in_order(
        cpu_body,
        [
            "for (int opIndex = 0; opIndex < thread.opCount; opIndex++) {",
            "const ThreadOp& op = thread.ops[opIndex];",
            "if (op.kind == THREAD_OP_STORE) {",
            "cpu_store_value(&ctx.testLocations[addr], op.order, 1);",
            "} else if (op.kind == THREAD_OP_LOAD) {",
            "uint value = cpu_load_value(&ctx.testLocations[addr], op.order);",
            "write_result_slot(&ctx.readResults[instance], op.resultSlot, value);",
        ],
        "execute_cpu_thread",
    )
    if "cuda::atomic_thread_fence(cuda::memory_order_seq_cst);" not in cpu_body:
        fail("execute_cpu_thread is missing the post-load seq_cst fence")

    gpu_body = extract_function_body(source_text, "__device__ void execute_gpu_thread(")
    ensure_in_order(
        gpu_body,
        [
            "for (int opIndex = 0; opIndex < thread.opCount; opIndex++) {",
            "const ThreadOp& op = thread.ops[opIndex];",
            "if (op.kind == THREAD_OP_STORE) {",
            "gpu_store_value(&testLocations[addr], thread, op.order, 1);",
            "} else if (op.kind == THREAD_OP_LOAD) {",
            "uint value = gpu_load_value(&testLocations[addr], thread, op.order);",
            "write_result_slot(&readResults[instance], op.resultSlot, value);",
        ],
        "execute_gpu_thread",
    )
    if "cuda::atomic_thread_fence(cuda::memory_order_seq_cst);" not in gpu_body:
        fail("execute_gpu_thread is missing the post-load seq_cst fence")


def build_verification_binary(
    root: Path,
    nvcc: str,
    mem_def: str,
    gpu_arch: str,
    host_arch_flags: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        nvcc,
        f"-D{mem_def}",
        f"-I{root}",
        "-rdc=true",
        "-arch",
        gpu_arch,
        "-Xcompiler",
        host_arch_flags,
        str(root / "ismm_runner.cu"),
        "-o",
        str(output_path),
        "-diag-suppress",
        "177",
    ]
    run_command(command, root)


def verify_host_disassembly(objdump: str, binary_path: Path, root: Path) -> None:
    store_text = run_command(
        [
            objdump,
            "-d",
            "-C",
            "--disassemble=cpu_store_value(het_atomic_uint*, MemOrderKind, unsigned int)",
            str(binary_path),
        ],
        root,
    )
    if "stlr" not in store_text:
        fail("cpu_store_value is missing stlr for release stores")
    if not re.search(r"mov\s+w2, #0x0.*?::store", store_text, re.S):
        fail("cpu_store_value is missing the relaxed store path")

    load_text = run_command(
        [
            objdump,
            "-d",
            "-C",
            "--disassemble=cpu_load_value(het_atomic_uint const*, MemOrderKind)",
            str(binary_path),
        ],
        root,
    )
    if "ldar" not in load_text:
        fail("cpu_load_value is missing ldar for rcsc loads")
    if "ldapr" not in load_text:
        fail("cpu_load_value is missing ldapr for rcpc loads")
    if not re.search(r"mov\s+w1, #0x0.*?::load", load_text, re.S):
        fail("cpu_load_value is missing the relaxed load path")
    if not re.search(r"mov\s+w1, #0x2.*?::load", load_text, re.S):
        fail("cpu_load_value is missing the acquire load path")


def verify_gpu_disassembly(cuobjdump: str, binary_path: Path, root: Path) -> None:
    sass_text = run_command(
        [cuobjdump, "--dump-sass", str(binary_path)],
        root,
    )
    required_patterns = {
        "relaxed loads": r"\bLDG\.E\b",
        "relaxed stores": r"\bSTG\.E\b",
        "system acquire loads": r"\bLD\.E\.STRONG\.SYS\b",
        "device acquire loads": r"\bLD\.E\.STRONG\.GPU\b",
        "system release stores": r"\bST\.E\.STRONG\.SYS\b",
        "device release stores": r"\bST\.E\.STRONG\.GPU\b",
        "seq_cst fence": r"\bMEMBAR\.SC\.SYS\b",
    }
    for label, pattern in required_patterns.items():
        if not re.search(pattern, sass_text):
            fail(f"cuobjdump SASS is missing {label}")

    sys_fence = sass_text.find("MEMBAR.ALL.SYS")
    sys_store = sass_text.find("ST.E.STRONG.SYS")
    gpu_fence = sass_text.find("MEMBAR.ALL.GPU")
    gpu_store = sass_text.find("ST.E.STRONG.GPU")
    if sys_fence < 0 or sys_store < 0 or sys_fence > sys_store:
        fail("system-scope release store sequence is missing MEMBAR.ALL.SYS before ST.E.STRONG.SYS")
    if gpu_fence < 0 or gpu_store < 0 or gpu_fence > gpu_store:
        fail("device-scope release store sequence is missing MEMBAR.ALL.GPU before ST.E.STRONG.GPU")


def extract_section_lines(text: str, start_marker: str, end_marker: str | None = None) -> tuple[str, ...]:
    start = text.find(start_marker)
    if start < 0:
        fail(f"missing assembly marker: {start_marker}")
    if end_marker is None:
        end = len(text)
    else:
        end = text.find(end_marker, start)
        if end < 0:
            fail(f"missing assembly end marker after {start_marker}: {end_marker}")
    section = text[start:end]
    lines = tuple(line for line in section.splitlines() if line.strip())
    if not lines:
        fail(f"empty assembly section for {start_marker}")
    return lines


def extract_matching_lines(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    lines = tuple(
        line
        for line in text.splitlines()
        if any(re.search(pattern, line) for pattern in patterns)
    )
    if not lines:
        fail(f"no assembly lines matched patterns: {patterns}")
    return lines


def extract_helper_asm(objdump: str, cuobjdump: str, host_binary: Path, gpu_binary: Path, root: Path) -> HelperAsm:
    host_store_text = run_command(
        [
            objdump,
            "-d",
            "-C",
            "--disassemble=cpu_store_value(het_atomic_uint*, MemOrderKind, unsigned int)",
            str(host_binary),
        ],
        root,
    )
    host_load_text = run_command(
        [
            objdump,
            "-d",
            "-C",
            "--disassemble=cpu_load_value(het_atomic_uint const*, MemOrderKind)",
            str(host_binary),
        ],
        root,
    )
    host_atomic_helpers = run_command(
        [objdump, "-d", "-C", "--start-address=0xeb08", "--stop-address=0xec30", str(host_binary)],
        root,
    )
    gpu_sass_text = run_command([cuobjdump, "--dump-sass", str(gpu_binary)], root)

    return HelperAsm(
        cpu_store_relaxed=extract_section_lines(
            host_store_text,
            "e65c:",
            "e67c:",
        ) + extract_section_lines(
            host_atomic_helpers,
            "000000000000eb08 <",
            "000000000000eb38 <",
        ),
        cpu_store_release=extract_section_lines(host_store_text, "e67c:", "e694:"),
        cpu_load_relaxed=extract_section_lines(host_load_text, "e71c:", "e738:") + extract_section_lines(
            host_atomic_helpers,
            "000000000000eb98 <",
            "000000000000ebbc <",
        ),
        cpu_load_acquire=extract_section_lines(host_load_text, "e738:", "e754:") + extract_section_lines(
            host_atomic_helpers,
            "000000000000eb98 <",
            "000000000000ebbc <",
        ),
        cpu_load_rcsc=extract_section_lines(host_load_text, "e754:", "e76c:"),
        cpu_load_rcpc=extract_section_lines(host_load_text, "e76c:", "e784:"),
        gpu_load_relaxed=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*04e0\*/\s+LDG\.E",
                r"/\*0520\*/\s+LDG\.E",
                r"/\*0540\*/\s+LDG\.E",
                r"/\*0580\*/\s+LDG\.E",
                r"/\*05c0\*/\s+LDG\.E",
                r"/\*0660\*/\s+LDG\.E",
                r"/\*0690\*/\s+LDG\.E",
                r"/\*06d0\*/\s+LDG\.E",
                r"/\*0700\*/\s+LDG\.E",
            ),
        ),
        gpu_load_acquire_system=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*0bf0\*/.*LD\.E\.STRONG\.SYS",
                r"/\*0c00\*/.*LD\.E\.STRONG\.SYS",
            ),
        ),
        gpu_load_acquire_device=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*0c50\*/.*LD\.E\.STRONG\.GPU",
                r"/\*0c60\*/.*LD\.E\.STRONG\.GPU",
            ),
        ),
        gpu_store_relaxed=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*0d30\*/\s+STG\.E",
                r"/\*0d50\*/\s+STG\.E",
                r"/\*0db0\*/\s+STG\.E",
                r"/\*0dd0\*/\s+STG\.E",
            ),
        ),
        gpu_store_release_system=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*0ed0\*/\s+MEMBAR\.ALL\.SYS",
                r"/\*0ef0\*/\s+ST\.E\.STRONG\.SYS",
                r"/\*0f20\*/\s+ST\.E\.STRONG\.SYS",
            ),
        ),
        gpu_store_release_device=extract_matching_lines(
            gpu_sass_text,
            (
                r"/\*0fc0\*/\s+MEMBAR\.ALL\.GPU",
                r"/\*0fe0\*/\s+ST\.E\.STRONG\.GPU",
                r"/\*1010\*/\s+ST\.E\.STRONG\.GPU",
            ),
        ),
    )


def helper_lines_for_op(op: OpSpec, thread: ThreadSpec, helper_asm: HelperAsm) -> tuple[str, ...]:
    if thread.domain == "cpu":
        if op.kind == "store":
            if op.order == "relaxed":
                return helper_asm.cpu_store_relaxed
            if op.order == "release":
                return helper_asm.cpu_store_release
        elif op.kind == "load":
            if op.order == "relaxed":
                return helper_asm.cpu_load_relaxed
            if op.order == "acquire":
                return helper_asm.cpu_load_acquire
            if op.order == "rcsc":
                return helper_asm.cpu_load_rcsc
            if op.order == "rcpc":
                return helper_asm.cpu_load_rcpc
    elif thread.domain == "gpu":
        if op.kind == "store":
            if op.order == "relaxed":
                return helper_asm.gpu_store_relaxed
            if op.order == "release" and thread.scope == "system":
                return helper_asm.gpu_store_release_system
            if op.order == "release" and thread.scope == "device":
                return helper_asm.gpu_store_release_device
        elif op.kind == "load":
            if op.order == "relaxed":
                return helper_asm.gpu_load_relaxed
            if op.order == "acquire" and thread.scope == "system":
                return helper_asm.gpu_load_acquire_system
            if op.order == "acquire" and thread.scope == "device":
                return helper_asm.gpu_load_acquire_device
    fail(f"no helper assembly mapping for thread={thread} op={op}")


def op_description(op: OpSpec) -> str:
    if op.kind == "store":
        return f"store {op.location} {op.order}"
    return f"load {op.location} {op.order} -> r{op.result_slot}"


def print_variant_assembly(doc_specs: dict[str, ExperimentSpec], helper_asm: HelperAsm) -> None:
    print("\nPer-Variant Memory Operations")
    print("=============================")
    for name in sorted(doc_specs):
        spec = doc_specs[name]
        print(f"\n{name} [{spec.test_kind}, expected={spec.expectation}]")
        for thread_index, thread in enumerate(spec.threads):
            print(f"  thread {thread_index}: domain={thread.domain}, scope={thread.scope}")
            for op_index, op in enumerate(thread.ops):
                print(f"    op{op_index}: {op_description(op)}")
                for asm_line in helper_lines_for_op(op, thread, helper_asm):
                    print(f"      {asm_line}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    docs_path = root / "docs" / "ISMM_TESTS.md"
    runner_path = root / "ismm_runner.cu"
    target_dir = root / "target"
    host_binary = target_dir / "ismm-runner-verify"
    cuobjdump_binary = target_dir / f"ismm-runner-verify-{args.cuobjdump_gpu_arch}"

    doc_text = docs_path.read_text()
    source_text = runner_path.read_text()

    doc_specs = parse_doc_experiments(doc_text)
    source_specs = parse_source_experiments(source_text)
    compare_experiments(doc_specs, source_specs)
    print(f"[ok] {len(doc_specs)} experiments match docs and source")

    verify_generic_execution_order(source_text)
    print("[ok] execute_cpu_thread and execute_gpu_thread preserve thread.ops order")

    build_verification_binary(
        root=root,
        nvcc=args.nvcc,
        mem_def=args.mem_def,
        gpu_arch=args.host_gpu_arch,
        host_arch_flags=args.host_arch_flags,
        output_path=host_binary,
    )
    print(f"[ok] built host verification binary: {host_binary.name}")

    build_verification_binary(
        root=root,
        nvcc=args.nvcc,
        mem_def=args.mem_def,
        gpu_arch=args.cuobjdump_gpu_arch,
        host_arch_flags=args.host_arch_flags,
        output_path=cuobjdump_binary,
    )
    print(f"[ok] built cuobjdump helper binary: {cuobjdump_binary.name}")

    verify_host_disassembly(args.objdump, host_binary, root)
    print("[ok] objdump verified ARM relaxed/release/LDAR/LDAPR paths")

    verify_gpu_disassembly(args.cuobjdump, cuobjdump_binary, root)
    print("[ok] cuobjdump verified GPU relaxed/acquire/release paths and scopes")

    helper_asm = extract_helper_asm(args.objdump, args.cuobjdump, host_binary, cuobjdump_binary, root)
    print_variant_assembly(doc_specs, helper_asm)

    print(
        "[ok] ISMM verification passed for all variants: docs/source experiment tables match, "
        "generic execution preserves per-thread op order, and emitted host/device instruction forms are present"
    )


if __name__ == "__main__":
    main()
