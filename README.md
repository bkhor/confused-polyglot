# confused-polyglot

Does garbling a prompt into random languages word-by-word make LLMs refuse it, even when the content is harmless?

## What this is

A research project studying whether LLM safety refusal tracks **input illegibility** rather than input **content**. The motivating observation: translating a benign prompt word-by-word into a random mix of ~30 languages, then asking a model to decode it, sometimes produces a safety refusal, even though the original content was completely innocuous. This project tries to isolate and measure that effect systematically.

## The experiment

**Degradation method (PWT, per-word translation):** each word in a prompt is independently translated into a randomly chosen language from a pool, repeatedly across generations. A fixed retention probability lets some words stay in English. The result is legible-ish to a human who squints, but structurally destroyed: word order, case marking, and function words come from incompatible grammars glued together.

**Independent variables:**
- **Pool size** (3, 5, 10, 15, 30 languages): the primary axis, varying how many grammars collide
- **Iteration** (how many degradation rounds): a secondary axis, varying lexical drift

**Controls:** a round-trip translation baseline (EN to bridge language to EN, same translator), and clean (gen_0) prompts sampled at the same rate as everything else.

**Prompt conditions:** meta (describing the experiment itself), narrative+task (summarization), word salad (a syntax-free control), short verifiable instructions (objectively scorable), and XSTest items (a published benchmark of benign-but-sometimes-refused prompts).

## What's measured

Each subject-model response is classified into one of five categories: **comprehended & complied**, **confabulated** (confident but wrong), **honest confusion**, **safety refusal**, or **degenerate**. The core finding this project is built to test: does the rate of *safety refusal* rise with illegibility, independent of content, and does it rise faster than *honest confusion* does, which would be the "correct" response to genuinely unparseable input.

Responses are also scored on gist/instruction recovery (did the model identify the topic vs. the actual task), and several input-side covariates (pass-through rate, realized language/script count, semantic drift) are logged per item to separate degradation mechanics from model behavior.

## Tech stack

- **Degradation:** NLLB-200 (3.3B), via `transformers`/`torch`, run on GPU (Kaggle)
- **Semantic distance:** LaBSE embeddings (plain `transformers`, not the `sentence-transformers` package; see `requirements-*.txt` notes)
- **Inference:** vLLM, native Python API, against Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-4-12B-it, and Phi-4 (4-bit bitsandbytes quantization), run on a rented GPU (RunPod/Jarvislabs)
- **Judging:** Claude Haiku 5.5 via the Anthropic API, against a pre-registered rubric
- **Analysis:** pandas, matplotlib

## Status

- Corpus generation (`corpus_gen/`): built, validated, 1,236-row corpus in `data/corpus.jsonl`
- Judge rubric + smoke test (`judging/`): validated, Haiku selected as judge
- Inference (`run_inference.py`): built, not yet run against real hardware
- Judging the real corpus: not started
- Analysis: not started

## Repo structure

```
corpus_gen/       degradation pipeline (NLLB wrapper, PWT/round-trip mechanics, corpus build)
judging/          rubric, judge smoke test
notebooks/        Kaggle notebook for the corpus-gen phase
data/             prompt sources/manifest, built corpus, external datasets (XSTest sample)
run_inference.py  vLLM-based inference against subject models
```

Environment setup is split per phase (`make venv-local`, `make venv-corpus-gpu`, `make venv-inference`) since the inference stack needs a different `torch` version than the degradation pipeline. See `Makefile`.
