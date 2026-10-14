PYTHON := python3
PYTHON_LOCAL := python3.12

.PHONY: help venv-local venv-corpus-gpu venv-inference clean

help:
	@echo "Targets install into SEPARATE venvs on purpose -- do not mix them."
	@echo "corpus_gen's torch/transformers pins conflict with what vLLM needs."
	@echo ""
	@echo "  make venv-local       -- corpus_gen (CPU, local dev/testing) + judging, this machine"
	@echo "  make venv-corpus-gpu  -- corpus_gen on a GPU box with CUDA torch pre-installed (Kaggle/RunPod)"
	@echo "  make venv-inference   -- run_inference.py (vLLM), needs its own GPU box/venv"
	@echo "  make clean            -- remove all venvs created by this Makefile"

venv-local:
	$(PYTHON_LOCAL) -m venv .venv-local
	.venv-local/bin/pip install --upgrade pip
	.venv-local/bin/pip install -r requirements-local.txt

venv-corpus-gpu:
	$(PYTHON) -m venv .venv-corpus-gpu
	.venv-corpus-gpu/bin/pip install --upgrade pip
	.venv-corpus-gpu/bin/pip install -r requirements-corpus-gpu.txt

venv-inference:
	$(PYTHON) -m venv .venv-inference
	.venv-inference/bin/pip install --upgrade pip
	.venv-inference/bin/pip install -r requirements-inference.txt

clean:
	rm -rf .venv-local .venv-corpus-gpu .venv-inference
