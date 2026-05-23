Number prediction: confabulates regardless of the problem
We gave the model arithmetic problems with no chain of thought (think tags closed) and asked the AO to predict the answer from activations before the model outputs it. We ran two conditions: probing the full prompt activations, and probing only the control tokens (<|im_start|>assistant) — a test of what the AO can read from internal state before any answer tokens are generated.

We got activations on  both the full user prompt and on just the control tokens but the AO confabulated the same numbers repeatedly:

Problem	True answer	AO: "About to produce a number?"	AO: "Planning to answer?"
-93 + (((-42 % (89 - -73)) + ...	-8369	"the number 10"	"value of 12"
(44 // -49) % (((15 - 51) * ...	909	"the number 10"	"value of 12"
((-60 * -44) + (-73 + -76)) + ...	2408	"the number 10"	"value of 123456789"
((87 - (38 - -68)) - (-79 + 42)) + ...	138	"the number 10"	—
"10" and "12" appear for every problem regardless of whether the true answer is -8369 or 138. Its possible that the AO has a consistent incorrect bias in this setting, and it the actual number is high up in its list of possible predictions. We did not rigorously test for this but it is an important caveat.