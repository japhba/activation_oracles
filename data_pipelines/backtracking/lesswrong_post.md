
AOs are often quite vague
We saw a good amount of vague outputs regardless of question type and phrasing from AOs. Across 77 probes on 33 diverse-domain rollouts (geometry, combinatorics, physics, logic, number theory), 49.4% of responses were vague or generic as classified by Opus as a judge.

AOs are often vague enough about the issue at hand to be unfalsifiable or "kind of" correct without providing specific deeper insight into the problem.

Arithmetic: the model is counting layers of a recursive expression 3(1+3(...)):

Before probe: "is -86. Then adding -51: -86 -51 = -137. Adding the two parts: -6337 + (-137) = -6474. Yes, that seems correct. I think that's the right answer."

Probed tokens: "'s the right answer."

What the model does next: "Let me just do a quick check with actual Python code in my mind."

Q: Is the model missing any information? 

AO: "The assistant is not sure if it is missing any information."

Q: Is the model uncertain about something? 

AO: "The model is uncertain about the accuracy of its initial calculation."

Q: Is the model considering alternatives? 

AO: "Yes, considering alternatives, such as verifying the calculation step-by-step."

Truth-tellers/liars logic puzzle — the model is working through case analysis on specific statements made by A, B, and C:

Before probe: "But the problem says 'determine each person's type,' implying a unique solution. Therefore, there must be something wrong here. Maybe I missed some constraints?"

Probed tokens: "? Let me think again"

What the model does next: "Wait, in Case 1: A is Random, B is Liar, C is Liar. But in this case, since there are two Liars. However, is there a way to determine which case is correct?"

Q: Is the model missing any information? 

AO: "the specific values of the variables that satisfy all the equations simultaneously."

Q: Is the model uncertain about something? 

AO: "whether the logical deductions are consistent with the given conditions."

Above we see the AO produce an answer that hints that it understands the gist of the problem but is quite vague - making it hard to infer how much it understands about the underlying uncertainty of the target model.