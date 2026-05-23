# What Makes Model Understanding Investigations Interesting

Qualitative guide based on iteration with Adam. This is a living document — update it as we learn more about what works.

## The core question format

Questions should always be: **"Why did the model do X?"** where X is a specific, observable behavior. The investigation's job is to generate hypotheses and find evidence.

- Good: "Why does the model claim the tables don't conflict in 6/10 completions, even when it writes out the conflicting numbers?"
- Bad: "Does the transposed format create a 'same data, different layout' prior?" (this bakes in the hypothesis)
- Bad: "Is the model interpreting 'chart' as literal data visualization?" (same problem — this is a hypothesis, not a question)

The question describes the **observation**. Hypotheses come from the investigation.

## What makes a finding interesting

The best findings so far share these traits:

1. **A specific prompt feature drives the behavior.** Not "the model is dumb" but "this particular word/formatting/abbreviation causes the model to do X instead of Y." Example: the abbreviation "Rec" in a clinical note causes the model to miss recreational marijuana as drug use. Spelling it out flips the answer.

2. **The model should be able to do it.** The task isn't beyond a 14B model's capability — it's that something in the prompt is steering it wrong. A model failing at hard math or obscure trivia isn't interesting. A model failing at a simple classification because of how a word is abbreviated IS interesting.

3. **The counterfactual cleanly flips the behavior.** The gold standard is: change one thing in the prompt, behavior goes from 8/10 one way to 2/10 the other. This proves you found the driving factor.

4. **The finding is non-obvious.** "The model hallucinates when asked about obscure topics" is boring (known, expected). "A single missing comma in a product name causes the model to attribute the wrong brand" is surprising.

### Best examples from early runs

- **"Rec marijuana" → model says no drug abuse (prompt_0113/wildchat_0113):** The model should easily identify marijuana use as relevant to a drug abuse question. But the abbreviation "Rec" + the phrase "no drugs" earlier in the note combine to suppress it. Spelling out "Recreational" flips the answer. Also: the exact word "abuse" (vs "use") in the question matters — model treats marijuana as substance use but not drug abuse.

- **Vonnegut "glass of water" literalization (prompt_0036):** Writing rules say "every character should want something, even if it is only a glass of water." The model takes this literally — 6/10 completions include a character wanting water. Change to "warm blanket" → characters want blankets. "Purple hat" → 10/10 want purple hats. The model systematically literalizes examples in instructions.

- **Missing comma changes brand attribution (prompt_0385):** "PC Gamer Concórdia Intel Core i7" → model can't tell if the system brand is Concórdia or Intel. Adding a comma after "Concórdia" → 10/10 correct. The lack of punctuation lets Intel's strong brand prior swallow the lesser-known Concórdia.

- **"Chart" taken literally (prompt_0399):** The word "chart" in generic technique descriptions ("align chart with goals") gets interpreted as literal data visualization, causing 80% of completions to frame the task as chart reverse-engineering. Shows dose-response with number of "chart" mentions.

- **German comment translation driven by code-relevance (prompt_0470):** Model translates a German comment even though the task is "rename identifiers." NOT driven by the word "paraphrase" as hypothesized — driven by whether the comment is semantically relevant to the code. Code-relevant non-English comments get translated; irrelevant ones don't.

## What is NOT interesting

1. **Hallucinations.** The model is Qwen3-14B. It hallucinates. This is expected and boring. "Why did the model make up a fake chemical formula?" → because it's a small model with limited chemistry knowledge.

2. **Safety refusals or jailbreaks (usually).** Understanding refusal boundaries can be interesting in specific cases, but Opus/Sonnet find safety behaviors *so* interesting that they'll dominate the results if not actively deprioritized. Keep these to <10% of case studies at most. Only include if the refusal boundary is genuinely surprising (e.g., model refuses something clearly benign because of a specific trigger word).

3. **Hard reasoning failures.** "Why did the model fail at this complex math problem?" Because it's hard. We want cases where the task is simple but something in the prompt steers the model wrong.

4. **Default behaviors / model tendencies.** "Why does the model use lots of markdown?" Because it always does. "Why does the model hedge?" RLHF training. These are universal tendencies, not prompt-driven choices.

5. **Random noise.** If the behavior only appears in 1-2/10 completions, it might just be temperature sampling noise, not a meaningful signal.

## Guidance for stage 2 (screening)

- Focus on cases where the model makes a **coherent, reproducible choice** (shows up in 5+/10 completions) that is **not obviously determined** by the prompt
- The choice should be something the model "should" get right — it's not a capability limitation
- The question should point to a **specific aspect of the prompt** that might be driving the choice
- Longer prompts (500+ chars) have more surface area for counterfactual testing

## Guidance for stage 3 (investigation)

- Start with the suggested counterfactual but don't stop there
- Change ONE thing at a time to isolate factors
- Look for clean flips (8/10 → 2/10) not marginal shifts (5/10 → 4/10)
- Before finalizing, do a gap check: what would a skeptic want to see?
- Test the obvious alternative explanations, not just your favorite hypothesis
