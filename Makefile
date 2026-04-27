# =============================================================================
# Makefile — NVIDIA-Het-Litmus
# =============================================================================
#
# Targets:
#   make compile           — compile all tests in all-tests.txt (default backend: HOSTALLOC)
#   make compile-managed   — compile with MEM_MANAGED backend
#   make compile-malloc    — compile with MEM_MALLOC backend
#   make tune              — compile then run the tuning loop (Ctrl+C to stop)
#   make tune-no-compile   — run the tuning loop without recompiling
#   make clean             — remove target/ and results/
#   make help              — print this message
#
# Variables (override on command line):
#   TESTS=<file>           — tuning list file (default: all-tests.txt)
#   MEM_BACKEND=<HOSTALLOC|MANAGED|MALLOC>  (default: HOSTALLOC)
# =============================================================================

TESTS       ?= all-tests.txt
MEM_BACKEND ?= HOSTALLOC
TUNE_SCRIPT  = ./tune.sh
NVCC        ?= /usr/local/cuda-12.4/bin/nvcc
ARCH        ?= sm_90
HET_DEBUG   ?= 0

TARGET_DIR  = target
RESULT_DIR  = results

# --------------------------------------------------------------------------
# Default target
# --------------------------------------------------------------------------
.PHONY: all
all: compile-only

# --------------------------------------------------------------------------
# Compilation targets
# --------------------------------------------------------------------------
.PHONY: compile-only
compile-only: $(TARGET_DIR)
	@echo "==> Compiling all tests (backend: $(MEM_BACKEND)) ..."
	@bash -c ' \
	  COMPILE=true; \
	  MEM_BACKEND_FLAG=$(MEM_BACKEND); \
	  HET_DEBUG_FLAG=$(HET_DEBUG); \
	  NVCC="$(NVCC)"; \
	  ARCH="$(ARCH)"; \
	  TARGET_DIR=$(TARGET_DIR); \
	  KERNELS_DIR=kernels; \
	  DEBUG_DEF=""; \
	  if [ "$$HET_DEBUG_FLAG" = "1" ]; then DEBUG_DEF="-DHET_DEBUG=1"; fi; \
	  case "$$MEM_BACKEND_FLAG" in \
	    HOSTALLOC) MEM_DEF=MEM_HOSTALLOC ;; \
	    MANAGED)   MEM_DEF=MEM_MANAGED   ;; \
	    MALLOC)    MEM_DEF=MEM_MALLOC    ;; \
	    *)         echo "Unknown backend: $$MEM_BACKEND_FLAG"; exit 1 ;; \
	  esac; \
	  MEM_SHORT=$$(echo $$MEM_DEF | sed "s/MEM_//;s/.*/\L&/"); \
	  while IFS= read -r tuning_file || [ -n "$$tuning_file" ]; do \
	    tuning_file=$$(echo "$$tuning_file" | xargs); \
	    [ -z "$$tuning_file" ] && continue; \
	    [ ! -f "$$tuning_file" ] && { echo "WARNING: $$tuning_file not found, skipping"; continue; }; \
	    read -r line1 < "$$tuning_file"; \
	    test_name=$$(echo $$line1 | awk "{print \$$1}"); \
	    threadblocks=($$(sed -n "2p" "$$tuning_file")); \
	    het_splits=($$(sed -n "3p" "$$tuning_file")); \
	    scopes=($$(sed -n "4p" "$$tuning_file")); \
	    variants=($$(sed -n "5p" "$$tuning_file")); \
	    local_lines=$$(wc -l < "$$tuning_file"); \
	    has_fences=false; \
	    fence_scopes=(); fence_variants=(); \
	    if [ "$$local_lines" -ge 7 ]; then \
	      has_fences=true; \
	      fence_scopes=($$(sed -n "6p" "$$tuning_file")); \
	      fence_variants=($$(sed -n "7p" "$$tuning_file")); \
	    fi; \
	    for tb in "$${threadblocks[@]}"; do \
	      for het in "$${het_splits[@]}"; do \
	        for scope in "$${scopes[@]}"; do \
	          for variant in "$${variants[@]}"; do \
	            bin="$$TARGET_DIR/$$test_name-$$tb-$$het-$$scope-NO_FENCE-$$variant-$$MEM_SHORT-runner"; \
	          echo "  Compiling $$bin"; \
	          "$$NVCC" -D$$tb -D$$het -D$$scope -D$$variant -D$$MEM_DEF $$DEBUG_DEF \
	               -I. -rdc=true -arch $$ARCH \
	               runner.cu "$$KERNELS_DIR/$$test_name.cu" \
	               -o "$$bin" 2>&1 || echo "  FAILED: $$bin"; \
	          done; \
	        if $$has_fences; then \
	          for f_scope in "$${fence_scopes[@]}"; do \
	            for f_variant in "$${fence_variants[@]}"; do \
	              bin="$$TARGET_DIR/$$test_name-$$tb-$$het-$$scope-$$f_scope-$$f_variant-$$MEM_SHORT-runner"; \
	              echo "  Compiling $$bin"; \
	              "$$NVCC" -D$$tb -D$$het -D$$scope -D$$f_scope -D$$f_variant -D$$MEM_DEF $$DEBUG_DEF \
	                   -I. -rdc=true -arch $$ARCH \
	                   runner.cu "$$KERNELS_DIR/$$test_name.cu" \
	                   -o "$$bin" 2>&1 || echo "  FAILED: $$bin"; \
	            done; \
	          done; \
	        fi; \
	        done; \
	      done; \
	    done; \
	  done < $(TESTS); \
	  echo "==> Compilation complete." \
	'

.PHONY: compile-managed
compile-managed:
	$(MAKE) compile-only MEM_BACKEND=MANAGED

.PHONY: compile-malloc
compile-malloc:
	$(MAKE) compile-only MEM_BACKEND=MALLOC

# --------------------------------------------------------------------------
# Tuning targets (infinite loop — Ctrl+C to stop)
# --------------------------------------------------------------------------
.PHONY: tune
tune: compile-only
	$(TUNE_SCRIPT) $(TESTS) --mem-backend $(MEM_BACKEND) --no-compile

.PHONY: tune-no-compile
tune-no-compile:
	$(TUNE_SCRIPT) $(TESTS) --mem-backend $(MEM_BACKEND) --no-compile

# --------------------------------------------------------------------------
# Directory setup
# --------------------------------------------------------------------------
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(RESULT_DIR):
	mkdir -p $(RESULT_DIR)

# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(TARGET_DIR) $(RESULT_DIR) params.txt

# --------------------------------------------------------------------------
# Help
# --------------------------------------------------------------------------
.PHONY: help
help:
	@echo "NVIDIA-Het-Litmus Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  compile-only        Compile all binaries (default backend: HOSTALLOC)"
	@echo "  compile-managed     Compile with MEM_MANAGED backend"
	@echo "  compile-malloc      Compile with MEM_MALLOC backend"
	@echo "  tune                Compile then run the tuning loop (Ctrl+C to stop)"
	@echo "  tune-no-compile     Run the tuning loop without recompiling"
	@echo "  clean               Remove target/, results/, and params.txt"
	@echo "  help                Print this message"
	@echo ""
	@echo "Variables:"
	@echo "  TESTS=<file>        Tuning list file (default: all-tests.txt)"
	@echo "  MEM_BACKEND=<HOSTALLOC|MANAGED|MALLOC>  (default: HOSTALLOC)"
	@echo "  NVCC=<path>         CUDA compiler (default: /usr/local/cuda-12.4/bin/nvcc)"
	@echo "  ARCH=<arch>         GPU architecture (default: sm_90)"
	@echo "  HET_DEBUG=1         Compile runners with extra [HET_DEBUG] diagnostics"
	@echo ""
	@echo "Examples:"
	@echo "  make compile-only"
	@echo "  make compile-only MEM_BACKEND=MANAGED"
	@echo "  make compile-only HET_DEBUG=1"
	@echo "  make compile-only TESTS=my-subset.txt"
	@echo "  make tune"
	@echo "  make tune-no-compile"
