# Training-free sentence embeddings: Token Prepending on LLaMA-2-7B (MLX)

Implementation of "Token Prepending" (Fu et al., arXiv 2412.11556) on LLaMA-2-7B + PromptEOL,
run locally on Apple Silicon with MLX (8-bit, group size 64), plus two enhancements:
ReverseEOL + TP (forward and reversed views) and CP + TP (contrastive steering), and their combination.

## Test-set averages (STS12-16, STS-B, SICK-R; Spearman x100)
| Run | Description | Avg |
|---|---|---|
| r1 | PromptEOL baseline | 70.07 (paper 70.03) |
| r2 | early exit, no TP | 74.87 |
| r3 | TP (s=1, e=7) | 77.25 (paper 77.19) |
| run 4 | ReverseEOL | 78.68 |
| run 5 | ReverseEOL + TP | 79.34 |
| run 7 | CP | 76.14 |
| run 8 | CP + TP | 77.60 |
| run 10 | ReverseEOL + TP + CP + TP | 79.57 |

Paired-bootstrap comparisons (B=2000) are in `notes/*_reading.md` and `results/results.jsonl`.
Run 10 vs run 5 is read as "no resolvable difference".

## Layout
- `src/` MLX backend, TP intervention, CP steering, caches, STS evaluation, statistics
- `scripts/` pass runner, scoring, bootstrap and per-stage scripts
- `frozen_settings.json` settings frozen before any test pass
- `results/results.jsonl` every recorded result
- `notes/` mechanism notes, reproduction table, enhancement readings
- `env/requirements-pinned.txt` pinned dependencies (Python 3.11)

## Not included
Model weights, caches, datasets, and the reference clones (`ref/token_prepending`, `ref/cp_repo`) are not in this repo.
Obtain Meta Llama-2-7b-hf (licence acceptance required), convert it to 8-bit group-64 MLX,
and fetch the SentEval data separately. Scripts expect `models/`, `data/` and `ref/` next to `src/`.
