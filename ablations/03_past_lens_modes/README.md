# 03 — past_lens mode and corpus ablations (A-H)

All on multi-layer L21/22/23 base. Each ablation varies one past_lens knob:

| tag | swap | result |
|---|---|---|
| baseline | cot-v5 + on-policy w/ inject | +0.292 |
| A_offpolicy | system_prompt_prob = 0 (no sys-prompt inject) | +0.264 |
| B_pastonly | directions = ["past"] only | +0.277 |
| C_finefineweb | corpus = m-a-p/FineFineWeb | +0.262 |
| D_fineweb | corpus = HuggingFaceFW/fineweb | +0.240 |
| E_past_vllm | past direction with vLLM-generated text (single-direction) | +0.284 |
| F_future_corpus | future direction, predict literal corpus next k tokens (no vLLM) | +0.270 |
| G_future_vllm_noinject | future, on-policy continuation, no sys-prompt inject (single-direction) | +0.283 |
| H_future_vllm_inject | future, on-policy + sys-prompt injection (single-direction) | +0.281 |

Key finding: corpus matters most (cot-v5 > FineFineWeb > fineweb). Direction/policy/injection are minor (within noise).
