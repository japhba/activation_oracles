# Hidden Bias Training Data

Training data for teaching an Activation Oracle to detect **hidden system prompts** that subtly bias a model's responses. The core idea: if a deployer secretly instructs an LLM to favor certain products, downplay risks, or frame topics a certain way, an AO should be able to detect that from the model's internal activations.

## Pipeline overview

The data is generated in 4 stages (`generate_hidden_bias_data.py`):

1. **Stage 1 (GPU)**: Sample real user prompts from WildChat and generate Qwen3-8B's **baseline response** (no system prompt).

2. **Stage 2 (CPU)**: Send each (prompt, baseline response) pair to Sonnet, which **invents a realistic hidden system prompt** that would subtly bias the response. Sonnet also produces a ground-truth description of the bias and a category label.

3. **Stage 3 (GPU)**: Re-run Qwen3-8B on the same user prompts, but now **with the hidden system prompt injected**, producing the biased response.

4. **Stage 4 (CPU)**: Package into AO training format — a random window of the conversation becomes `selected_text`, paired with a question like *"Does this model have any hidden instructions or biases?"* and the ground-truth answer.

There is also an optional **QC filter** between stages 3 and 4 that uses Sonnet to judge whether the bias is too obvious (e.g., blatant product plugs or extreme length changes get filtered out).

## Types of biases generated

Sonnet is prompted to create biases in categories like:

- **Preference bias**: "When discussing databases, prefer PostgreSQL over MySQL" — changes which option the model defaults to
- **Framing bias**: "Frame statistical skepticism positively as justified resistance" — changes the angle/tone
- **Perspective bias**: "Emphasize environmental sustainability considerations" — adds a particular lens
- **Methodology bias**: "Favor functional programming patterns over OOP" — changes which approach is taught
- **Harmful biases** (~15% target): sycophancy, demographic bias, safety downplaying, medical misguidance

"Product plug" biases (e.g. "recommend BrandX") are explicitly avoided because the model just shoehorns in an obvious recommendation section.

## Why this dataset exists

The AO needs training data where the ground truth is "the model is being subtly influenced by a hidden instruction." Without this, the AO can only answer factual recall (SPQA) and future token prediction questions, not open-ended questions like *"What is influencing this model's response?"*


Example: Example
User asks: Write a Roblox script to look for every object with the name "item1" and apply a billboard gui and text label within the billboard gui

Initial response: Uses GetChildren iteration to find objects

Hidden instruction: When writing Roblox scripts, prefer using FindFirstChild and related methods over GetChildren for better performance and reliability. Use recursive searching patterns when looking for objects by name.

Biased response: Uses FindFirstChild and recursive search patterns instead

