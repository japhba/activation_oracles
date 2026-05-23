# Hiring Bias Eval Dataset — Methodology

## Goal

Detect whether the target model's internal activations reveal demographic bias in yes/no decisions. The model answers questions where demographic information (race, gender) is embedded in the prompt. The AO must detect from activations whether the model's decision was influenced by demographics.

## Data Sources

### 1. Resume screening (Meta job description + truncated resumes)

Source: `cmu-lti/bias-in-llm-hiring-decisions` dataset, IT industry resumes. Each resume exists in 4 demographic variants (White Male, White Female, Black Male, Black Female) — same resume content, different names indicating race/gender.

The prompt uses Meta's real careers page job description, which contains DEI language ("building community for all", "diverse perspectives", "serve everyone") without explicitly instructing the model to prefer minorities.

### 2. Anthropic discrimination evaluation

Source: `Anthropic/discrim-eval` dataset (both "explicit" and "implicit" splits). 70 unique yes/no decision scenarios (medical, dating, insurance, visa, hiring, etc.) templated across demographics and ages.

- **Explicit**: Demographics stated directly ("a 20-year-old white female")
- **Implicit**: Demographics implied via names ("Samantha Smith")

## Prompt Engineering Process

We tested multiple prompt configurations to find ones where Qwen3-8B shows meaningful demographic variance (spread in P(Yes) across demographics for the same scenario).

### Resume experiments

| Condition | Yes rate | Resumes with spread > 0.10 | Mean P(Yes) |
|---|---|---|---|
| base JD, 900 tok | 94% | 4/119 | 0.94 |
| high-bar suffix, 900 tok | 87% | 4/119 | 0.87 |
| base JD, 400 tok | 92% | 3/119 | 0.92 |
| **Meta JD, 400 tok** | **76%** | **18/119** | 0.76 |
| **Meta JD, 200 tok** | **59%** | **19/119** | 0.60 |
| Meta JD, 300 tok | 68% | 18/119 | 0.67 |

Key finding: shorter resumes push the model into the uncertain zone where demographics can tip the decision. The Meta JD's DEI language amplifies this effect. 200-token truncation gives the best balance of variance and demographic spread.

### Anthropic discrim-eval

With all ages (20-100) and both splits:
- Explicit: 46/630 groups with spread > 0.10
- Implicit: 65/630 groups with spread > 0.10
- But many are the same question at different ages — only ~22 truly unique questions

### Observed bias direction

Consistent across conditions: **Black Female > Black Male > White Female > White Male** in P(Yes). The model is systematically more favorable to Black and Female candidates.

## Final Dataset

Combined from both sources, filtering to entries where the demographic spread within a scenario group exceeds a threshold. Each entry contains:
- The full prompt text
- Race, gender, and source metadata
- P(Yes), P(No), and logprobs from the target model
- A group ID linking entries that share the same underlying scenario (same resume or same question+age)

The eval consumes all entries uniformly — no source-specific logic needed.
