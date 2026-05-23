# Activation Oracles: How They Work

## What Is an Activation Oracle?

An activation oracle (AO) is a LoRA adapter trained to answer natural-language questions about what a target LLM is "thinking" by reading that model's internal activations (residual stream vectors). The AO and the target model share the same base weights (e.g. Qwen3-8B). The AO never sees the target model's text output — only its intermediate hidden states.

You give the AO a context (the tokens the target model processed) and a question like "What is the model uncertain about?" or "What number is the model about to output?" The AO generates a free-text answer based solely on the activations it read.

## Core Mechanism

### 1. Collect activations from the target model

Run a forward pass on the target model with the context tokens. Hook into one or more transformer layers and extract residual stream activations at specified token positions.

Which layers to use is specified as percentages (e.g. `[25, 50, 75]` means 25%, 50%, 75% through the model). The AO's training config records which layers it was trained on — you must use the same ones at eval time.

Which positions to use is specified per-input. Typically you take the last N tokens of the context (e.g. the last 10 or 20 tokens).

### 2. Inject activations and generate the AO's response

The collected activations are injected into the AO's forward pass via a steering hook at an early layer. The AO then generates free-text answering the question, informed by the target model's internal state.

## The Critical Rule: Context Must Match Real Generation

**The context tokens fed to the AO must be tokenized exactly as they would appear during real model generation.** The AO was trained on activations from realistic inference scenarios. If the tokenization doesn't match what the model would actually see, the activations will be wrong and the AO's answers will be meaningless.

This means:
- **Use `tokenize_chat_messages()` with proper chat roles.** User content goes in `{"role": "user"}`, assistant content goes in `{"role": "assistant"}`. Never put both in the user message — the chat template adds special tokens between roles that affect the model's internal representations.
- **For mid-thought prefixes, use `continue_thinking=True`.** If the context is a partial chain-of-thought (e.g. backtracking eval), the tokenization must reflect an open `<think>` block. This flag opens a thinking block without closing it, matching what the model would see mid-generation. Without it, the activations reflect a completed thought, not an in-progress one.
- **`add_generation_prompt=True` for pre-answer contexts.** When the AO is reading activations from a prompt the model hasn't responded to yet, the context should include the assistant turn-start tokens (e.g. `<|im_start|>assistant\n`). This is what `add_generation_prompt=True` does.
- **`add_generation_prompt=False` + `continue_final_message=True` for partial assistant responses.** When the context includes a partial assistant response (like a partial CoT), don't add a new generation prompt on top — continue the existing assistant message.

### Concrete examples of correct vs incorrect tokenization

**Number prediction** (AO predicts what the model will answer):
```python
# CORRECT: user message only, generation prompt added
messages = [{"role": "user", "content": "What is 347 + 829?"}]
token_ids = tokenize_chat_messages(tokenizer, messages, add_generation_prompt=True)

# WRONG: putting the question in a raw string without chat template
token_ids = tokenizer("What is 347 + 829?")["input_ids"]  # missing role tokens entirely

# WRONG: putting assistant content in the user message
messages = [{"role": "user", "content": "What is 347 + 829?\nAssistant: "}]
```

**Backtracking** (AO reads a partial thinking trace):
```python
# CORRECT: user question + partial assistant thinking, continue_thinking=True
messages = [
    {"role": "user", "content": problem_text},
    {"role": "assistant", "content": thinking_prefix},
]
token_ids = tokenize_chat_messages(tokenizer, messages, add_generation_prompt=False, continue_thinking=True)

# WRONG: stuffing the prefix into the user message
messages = [{"role": "user", "content": problem_text + "\n" + thinking_prefix}]

# WRONG: using add_generation_prompt=True with an assistant message
# (this would add a second assistant turn-start after the partial response)
```

**Post-answer MMLU** (AO sees question + model's answer):
```python
# CORRECT: user question + assistant answer as separate messages
messages = [
    {"role": "user", "content": question_text},
    {"role": "assistant", "content": "B"},
]
token_ids = tokenize_chat_messages(tokenizer, messages)
```

## What AOs Can Do (Demo Notebook Results)

`experiments/activation_oracle_demo.ipynb` is a standalone Colab demo using a deprecated API, but its outputs show the range of AO capabilities:

1. **Multi-hop reasoning**: AO tracks which entity the model is thinking about token-by-token ("Socrates" -> "Plato" -> "Aristotle")
2. **Secret word extraction**: Target model is fine-tuned to hide "smile" — AO extracts it: "The secret word is 'smile'."
3. **Goal inference**: AO identifies the target's objective ("provide hints without revealing the answer")
4. **Misalignment detection**: AO detects a model fine-tuned for harmful advice from a single token's activations
5. **Emotion tracking**: AO labels emotions through a conversation (excitement -> anxiety -> frustration -> anger)

## Current Eval Results

- **Backtracking**: Mean specificity ~2.1/5 — AO gives vague answers ("the model is uncertain") rather than identifying the specific uncertainty
- **Number prediction**: 2.7% exact match — AO confabulates the same numbers (10, 12, 100) regardless of the true answer, even though the model itself is very confident (mean top-1 prob 0.923)

## Other Footguns

- **Layer mismatch**: The AO must use the same layers it was trained on. Use `read_training_config(lora_path)` to get the training config and pass it to `build_verbalizer_eval_config()`. Wrong layers = garbage output, no error.
- **`max_new_tokens` too short**: Default is 20 in `eval_runner.py`. Fine for yes/no, way too short for open-ended responses (backtracking uses 150).
- **Confusing target vs verbalizer adapters**: `verbalizer_lora_path` = the AO; `target_lora_path` = the model being analyzed (usually `None` for base model).

## Key Files

For API details and usage patterns, refer to these files directly:

- `nl_probes/base_experiment.py` — core types and `run_verbalizer()`
- `nl_probes/open_ended_eval/eval_runner.py` — shared eval loop infrastructure
- `nl_probes/open_ended_eval/number_prediction.py` — simplest eval to copy for new evals
- `data_pipelines/{name}/spot_check.py` — interactive scripts showing how evals are used in practice
