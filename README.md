# NusaSynth

NusaSynth is a multi-agent framework that generates synthetic sentiment data for low-resource Indonesian local languages. It expands the NusaX-Senti training set, which holds 500 sentences per language, while keeping the sentiment label and the language of each sentence in place.

Four LLM agents share the work. The Contextualizer analyzes each seed sentence and writes variation plans, the Generator turns the plans into candidate sentences, and two validators check sentiment and language. Each validator consults an external classifier as a grounding signal: a fine-tuned NusaBERT classifier for sentiment and GlotLID for language identification. A per-sentence length filter and cross-batch deduplication follow, and a rejected sentence returns to the Generator with a diagnostic reason, up to two retries.

This repository accompanies the paper *NusaSynth: A Multi-Agent Framework for Synthetic Sentiment Data Generation in Indonesian Low-Resource Languages* (ICoDiT 2026).

## Corpus

`corpus/` holds 16,590 synthetic sentences in seven languages, one CSV file per language.

| Language | File | Sentences |
|---|---|---:|
| Acehnese | `ace.csv` | 2,268 |
| Balinese | `ban.csv` | 2,400 |
| Banjarese | `bjn.csv` | 2,366 |
| Javanese | `jav.csv` | 2,430 |
| Madurese | `mad.csv` | 2,298 |
| Minangkabau | `min.csv` | 2,381 |
| Sundanese | `sun.csv` | 2,447 |
| **Total** | | **16,590** |

| Column | Description |
|---|---|
| `id` | Sentence index within the file |
| `seed_id` | `id` of the NusaX-Senti training sentence used as the seed |
| `text` | Synthetic sentence |
| `label` | `negative`, `neutral`, or `positive` |

All seeds come from the NusaX-Senti training split. The sentences were generated with Gemini 3.1 Flash-Lite and are meant to be added to the original training data.

## Code

| Path | Contents |
|---|---|
| `NusaSynth/` | The generation pipeline, built with LangGraph |
| `scripts/make_ratio_variants.py` | Builds the augmented training sets (seed + synthetic) |
| `scripts/finetune_nusabert_base.py`, `scripts/finetune_nusabert_large.py`, `scripts/finetune_xlmr.py` | Downstream fine-tuning of NusaBERT base/large and XLM-R base/large |
| `scripts/sv_grounding/p1_sweep/pissa_drop06_full.py` | Trains the NusaBERT-large classifiers that ground the Sentiment Validator (LoRA with PiSSA initialization, linear probing then fine-tuning) |
| `scripts/sv_grounding/p0_multiseed/p0_ft_multiseed.py` | Full fine-tuning reference for the grounding classifier |
| `scripts/ablation_dataquality.py`, `scripts/ablation_purity.py` | Data-quality comparison with the Single-LLM baseline |

Main files in `NusaSynth/`:

| File | Role |
|---|---|
| `run_pipeline.py` | Entry point: per-language loop and cross-batch deduplication |
| `graph.py` | Pipeline graph and feedback loop |
| `nodes.py` | Contextualizer, Generator, and the length filter |
| `sv_graph.py` | Sentiment Validator |
| `lv_graph.py` | Linguistic Validator |
| `prompts.py` | Agent prompts and output schemas |
| `tools.py` | Grounding models (NusaBERT, GlotLID) and similarity helpers |
| `config.py` | Model, temperature, batch size, and path settings |
| `single_llm.py` | Single-LLM baseline used for comparison |

## Usage

Requirements: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), a CUDA GPU for fine-tuning, and access to Gemini through Vertex AI.

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Download NusaX-Senti from https://github.com/IndoNLP/nusax into `data/nusax_senti/<lang>/{train,valid,test}.csv`.

3. Train the grounding classifiers. The script trains all twelve NusaX languages over five seeds and keeps every checkpoint (about 80 GB) in `outputs/pissa-drop06-full/`. The pipeline loads seed 42.

   ```bash
   uv run python scripts/sv_grounding/p1_sweep/pissa_drop06_full.py
   ```

4. Generate the corpus. GlotLID is downloaded from Hugging Face (`cis-lmu/glotlid`) on first use. Output goes to `outputs/synthetic/<run_id>/<lang>/synthetic.csv`, with a full audit log in `pipeline_log.jsonl`. Set `PIPELINE_MODE=single_llm` to run the Single-LLM baseline instead.

   ```bash
   export GOOGLE_CLOUD_PROJECT=<your-project-id>
   export GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account.json>
   uv run python -m NusaSynth.run_pipeline
   ```

5. Build the augmented training sets and fine-tune a downstream classifier on them. Each script runs seeds 42, 0, and 123 and evaluates on the NusaX-Senti test split. Use `XLMR_SIZE=base` or `XLMR_SIZE=large` to choose the XLM-R variant.

   ```bash
   uv run python scripts/make_ratio_variants.py
   uv run python scripts/finetune_nusabert_large.py
   ```

## License

The corpus in `corpus/` is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (see `corpus/LICENSE`). Its seed sentences come from NusaX-Senti, which is released under the same license. The code is licensed under Apache-2.0 (see `LICENSE`).

## Citation

The paper has been accepted at ICoDiT 2026, and citation details will be added after publication. The corpus builds on NusaX, so please also cite it:

```bibtex
@inproceedings{winata-etal-2023-nusax,
    title = "{N}usa{X}: Multilingual Parallel Sentiment Dataset for 10 {I}ndonesian Local Languages",
    author = "Winata, Genta Indra  and
      Aji, Alham Fikri  and
      Cahyawijaya, Samuel  and
      Mahendra, Rahmad  and
      Koto, Fajri  and
      Romadhony, Ade  and
      Kurniawan, Kemal  and
      Moeljadi, David  and
      Prasojo, Radityo Eko  and
      Fung, Pascale  and
      Baldwin, Timothy  and
      Lau, Jey Han  and
      Sennrich, Rico  and
      Ruder, Sebastian",
    editor = "Vlachos, Andreas  and
      Augenstein, Isabelle",
    booktitle = "Proceedings of the 17th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.57/",
    doi = "10.18653/v1/2023.eacl-main.57",
    pages = "815--834"
}
```
