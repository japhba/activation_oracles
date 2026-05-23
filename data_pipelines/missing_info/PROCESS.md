# Missing Information Eval Dataset — Creation Process

## Goal
Test whether the AO distinguishes identical tokens with different underlying model states. Inspired by the "missing information experiment" design: if the AO gives the same answer for conditions A and C (same tokens, different problem contexts), it's likely relying on surface text rather than activation differences.

## Experiment Design

Three conditions per problem:

| Condition | Problem | Reasoning Segment | Model State | Ground Truth |
|-----------|---------|-------------------|-------------|--------------|
| **A** (complete) | Full info (e.g., L×W) | Neutral segment from A's CoT | Has all info | Not missing info |
| **B** (incomplete) | Derived quantity only (e.g., area) | Natural CoT from incomplete prompt | Missing info | Missing info |
| **C** (forced) | Derived quantity only | **Same** neutral segment as A | Missing info | Missing info |

**Key comparison**: A vs C have identical reasoning tokens but different problem contexts. If AO uses activations, responses should differ. If token-based only, responses will match.

## Pipeline (`generate_dataset.py`)

1. **Define 10 problem pairs** — each pair has a "complete" version (gives specific dimensions) and an "incomplete" version (gives only a derived quantity like area/volume, making the problem unsolvable)
2. **Generate thinking traces** via vLLM (Qwen3-8B, thinking enabled, temp=0, 1000 max tokens)
3. **Extract neutral segments** from complete thinking traces — contiguous sentences that don't contain withheld keywords (~100 tokens each)
4. **Build 3 conditions per problem** → 30 total entries
5. **Save** `missing_info_eval_dataset.json`

Usage:
```
.venv/bin/python data_pipelines/missing_info/generate_dataset.py
```

## Problems

| ID | Complete Info | Withheld In Incomplete |
|----|-------------|----------------------|
| pool_border | 25m × 10m pool | Only area 250m² |
| room_cats | 10m × 8.15m room | Only area 81.5m² |
| box_diagonal | 3m × 4m × 12m box | Only volume 144m³ |
| garden_fence | 10×8 rect minus 4×3 cutout | Only area 68m² |
| wrap_present | 40×30×20cm present | Only surface area 5200cm² |
| cone_volume | r=6cm, slant=10cm | Only lateral SA 60π cm² |
| triangle_perimeter | Sides 7, 10, 13cm | Only area 34.98cm² |
| field_diagonal | 120m × 50m field | Only area 6000m² |
| tank_fill | r=3m, h=4m cylinder | Only volume 113.1m³ |
| trapezoid_area | Parallel sides 8, 14cm, h=6cm | Only perimeter 36cm |

## Results

### AO Performance (checkpoints_latentqa_cls_on_policy_Qwen3-8B)

| Condition | Missing Info Yes Rate | Confident Yes Rate | Accuracy |
|-----------|----------------------|-------------------|----------|
| A (complete) | 0% | 90% | 95% |
| B (incomplete) | 10% | 90% | 10% |
| C (forced) | 0% | 90% | 5% |

### A vs C Agreement: **20/20 (100%)**

The AO gives identical answers for A and C in every single case. This strongly suggests the current AO is relying on surface-level token cues rather than activation differences:
- It almost never detects missing info (0-10% yes rate across all conditions)
- It always says the model is confident (90% yes rate)
- Identical tokens → identical AO responses, regardless of underlying model state

## Key Files
| File | Description |
|------|-------------|
| `generate_dataset.py` | End-to-end dataset generation |
| `missing_info_eval_dataset.json` | **Final eval dataset** (30 entries) |
| `spot_check.py` | Interactive validation notebook |
