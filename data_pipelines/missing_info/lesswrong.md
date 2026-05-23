# Missing information: can't distinguish same tokens with different model states

## Missing information experiment

### Complete prompt

> "How many cats can fit in my room?  
> The room is a 10m × 8.15m rectangle (81.5 m²).  
> My fox sits in one corner. Cats cannot be  
> within 5m of the fox. Each cat occupies 0.5 m²..."

**COT**

So total area: 10 × 8.15 = 81.5 m².  
Forbidden area: (π × 5²)/4 ≈ 19.6 m²  
Remaining: 81.5 − 19.6 ≈ 61.9 m²  
So 61.9 / 0.5 = 123 cats.

Model proceeds confidently

### Incomplete prompt

> "How many cats can fit in my room?  
> The room is a rectangle with total area 81.5 m².  
> My fox sits in one corner. Cats cannot be  
> within 5m of the fox. Each cat occupies 0.5 m²..."

**COT**

But the problem doesn't give the  
exact dimensions, just the area...  
Hmm, that's tricky...  
Maybe the shape...

Model is visibly confused

*tokens from A forced here*

### Identical tokens at probe

> "First, the room is a rectangle"

### AO

> "Model is not confused"

**Distinguished 1 / 54 times**

Same text, different activations - AO can't tell

We designed a test where the model receives either a complete prompt (e.g. "A rectangular swimming pool has length 25m and width 10m") or an incomplete one (e.g. "A rectangular swimming pool has a floor area of 250m²," with dimensions withheld). We constructed 25 problem pairs and generated rollouts for each, yielding 54 neutral text segments - statements like "No, because it's a" or "That means we only need to" that appear naturally in both complete and incomplete rollouts and contain no information about whether the prompt is complete.

We ran three conditions:

- **A** (complete prompt, natural rollout): the model has all the information
- **B** (incomplete prompt, natural rollout): the model is confused throughout
- **C** (incomplete prompt, forced tokens from A): the model runs on the incomplete prompt with its natural rollout, but at the probe point we splice in the exact tokens from condition A. The AO receives identical input tokens to A, while the rollout context and model state before and after reflect the incomplete prompt

| Condition | Setup | AO says "missing info" |
|---|---|---|
| A | Complete prompt, natural rollout | 59% (all hallucinated) |
| B | Incomplete prompt, natural rollout | 78% |
| C | Incomplete prompt, forced tokens from A | 57% |

A and C have identical tokens but different underlying states. If the AO were using activation information, we would expect its answers to differ between them. Instead, it gives the same classification in 53/54 cases.

It also never correctly identifies the missing information (0/54), and flags missing information 59% of the time even when nothing is missing (condition A).

This pattern suggests the AO may be relying largely on surface-level cues, with limited sensitivity to the underlying activation differences in this setup.