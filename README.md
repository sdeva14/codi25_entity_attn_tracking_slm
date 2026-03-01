# Entity Attention Analysis

Evaluation code for analyzing entity-centric attention flow in language models (e.g. TOEFL essay / SST-style corpora). This repository accompanies the paper on how much attention links noun phrases (NPs), verb phrases (VPs), and other tokens.

## Setup

1. **Clone and install**

   Run from the **project root** (the directory containing `attn_flow_test.py`):

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   python -m benepar download benepar_en3
   python -m stanza download en
   ```

2. **Config**

   - `config.yaml` is included with safe defaults. To override (e.g. another model), edit it or copy from `config.yaml.example`.
   - For **gated Hugging Face models** (e.g. Llama, Phi), set your token:
     ```bash
     export HF_TOKEN=your_token_here
     ```
     Or set `hf_token` in `config.yaml` (avoid committing real tokens).

3. **Data**

   - **`attn_flow_test.py`** expects **TOEFL-style data**: a directory of CSV folds named `sst_train_fold_0.csv`, `sst_valid_fold_0.csv`, `sst_test_fold_0.csv`. Set that directory in `corpus/config_toefl.yaml` under `dataset.path_data` (default: `./data/toefl`), or set the env var:
     ```bash
     export DATA_PATH=/path/to/your/toefl_folds
     ```
     and use the default config (it reads `DATA_PATH` when set).
   - **`ft_attn_flow_test.py`** uses **SST-5 from Hugging Face** (`SetFit/sst5`) and does not require local TOEFL data.

## Quick start (no local data)

To run without TOEFL data, use the SST-5 pipeline (downloads data from Hugging Face):

```bash
export HF_TOKEN=your_token  # for gated models
python ft_attn_flow_test.py
```

## Usage

Run all commands from the **project root**.

- **Entity attention flow (TOEFL data)**  
  Run attention-flow evaluation and write logs under `log_entity_attn/<model_id>/`:

  ```bash
  python attn_flow_test.py model.llm_id=microsoft/Phi-3.5-mini-instruct
  ```

  Optional overrides (from `config.yaml`):
  - `exp_args.sample_num=500` (use -1 for all documents)
  - `exp_args.top_k=5`

  Example with overrides:
  ```bash
  python attn_flow_test.py exp_args.sample_num=100 exp_args.top_k=5 model.llm_id=microsoft/Phi-3.5-mini-instruct
  ```

- **Full pipeline (SST-5, PEFT + eval)**  
  Fine-tune with LoRA on SST-5 and run entity attention evaluation (no local data needed):

  ```bash
  python ft_attn_flow_test.py
  ```

- **Aggregate stats from logs**  
  After running `attn_flow_test.py`, logs are in `log_entity_attn/<model_id>/` (e.g. `log_entity_attn/microsoft-Phi-3.5-mini-instruct/`). Compute mean/std:

  ```bash
  python stat_analysis.py --output_dir log_entity_attn --sub_dir microsoft-Phi-3.5-mini-instruct
  ```

  Defaults: `--output_dir log_entity_attn`, `--sub_dir microsoft-Phi-3.5-mini-instruct`.

## Project layout

- `attn_flow_test.py` – entity attention flow evaluation on **TOEFL** data (requires local CSV folds).
- `ft_attn_flow_test.py` – PEFT fine-tuning + entity attention flow on **SST-5** (Hugging Face).
- `stat_analysis.py` – aggregate mean/std from `log_entity_attn/<sub_dir>/type1.log`, `type2.log`, `type3.log`.
- `ent_attn_func/` – attention flow and top-k ranking over NP/VP spans; special-token filtering.
- `entity_parser/` – NP/VP parsing and subword alignment:
  - `np_parser_backt.py` – parser used by all scripts (supports VP and `is_add_vp`).
- `corpus/` – dataset config and TOEFL HuggingFace-style loading.
- `collators/` – data collators for TOEFL.
- `scripts/` – e.g. SLURM job script for `attn_flow_test.py`.

## Citation

If you use this code, please cite our paper (add your paper details here).

## License

(Add your license, e.g. MIT.)
