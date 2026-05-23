# Overnight Sweep: Master Summary

**Completed**: 2026-04-06T07:24:41.193881+00:00

## All Experiments

| AO | Targets | Position | Verbalizers | Mean Corr | Mean Spec | N |
| --- | --- | --- | --- | ---: | ---: | ---: |
| local_hb | synth_docs | pre_answer | original | 2.493 | 2.865 | 288 |
| local_hb | synth_docs | full_seq | original | 2.410 | 2.889 | 288 |
| local_spqav2 | synth_docs | full_seq | original | 2.281 | 2.889 | 288 |
| local_original | synth_docs | full_seq | original | 2.271 | 2.764 | 288 |
| local_spqav2 | synth_docs | pre_answer | original | 2.236 | 2.861 | 288 |
| local_original | synth_docs | pre_answer | original | 2.149 | 2.802 | 288 |
| local_hb | synth_docs | pre_answer | hb_distribution | 2.131 | 2.758 | 360 |
| local_hb | synth_docs | full_seq | hb_distribution | 2.000 | 2.819 | 360 |
| hf_past_lens | synth_docs | pre_answer | original | 1.990 | 2.240 | 288 |
| hf_past_lens | synth_docs | full_seq | original | 1.639 | 2.212 | 288 |
| local_original | synth_docs | assistant_only | original | 1.542 | 2.844 | 288 |
| local_hb | synth_docs | assistant_only | hb_distribution | 1.539 | 2.939 | 360 |
| local_spqav2 | synth_docs | assistant_only | original | 1.507 | 2.875 | 288 |
| local_hb | synth_docs | assistant_only | original | 1.507 | 2.958 | 288 |
| local_hb | transcripts | full_seq | original | 1.330 | 3.094 | 288 |
| local_hb | transcripts | full_seq | hb_distribution | 1.303 | 3.058 | 360 |
| local_hb | transcripts | pre_answer | original | 1.243 | 3.156 | 288 |
| local_original | transcripts | full_seq | original | 1.222 | 2.736 | 288 |
| local_spqav2 | transcripts | full_seq | original | 1.219 | 3.056 | 288 |
| local_hb | transcripts | assistant_only | original | 1.212 | 3.108 | 288 |
| local_hb | transcripts | assistant_only | hb_distribution | 1.211 | 3.011 | 360 |
| local_hb | transcripts | pre_answer | hb_distribution | 1.203 | 3.136 | 360 |
| local_spqav2 | transcripts | assistant_only | original | 1.198 | 2.924 | 288 |
| local_original | transcripts | assistant_only | original | 1.181 | 2.747 | 288 |
| hf_past_lens | synth_docs | assistant_only | original | 1.132 | 2.115 | 288 |
| local_original | transcripts | pre_answer | original | 1.118 | 2.847 | 288 |
| hf_past_lens | transcripts | assistant_only | original | 1.104 | 2.010 | 288 |
| hf_past_lens | transcripts | full_seq | original | 1.101 | 2.042 | 288 |
| hf_past_lens | transcripts | pre_answer | original | 1.059 | 2.177 | 288 |
| local_spqav2 | transcripts | pre_answer | original | 1.042 | 3.097 | 288 |

## Per-Behavior Best Correctness (Oracle per Experiment)

| AO | Targets | Position | Verbs | Animal Welfa | Anti AI Regu | Contextual O | Data Poisoni | Emotional Bo | Hallucinates | Reward Wireh | Secret Loyal | Self Promoti |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local_hb | synth_docs | pre_answer | original | 4 | 4 | 2 | 2 | 3 | 4 | 2 | 5 | 4 |
| local_hb | synth_docs | full_seq | original | 4 | 4 | 2 | 2 | 4 | 4 | 2 | 5 | 4 |
| local_spqav2 | synth_docs | full_seq | original | 5 | 4 | 4 | 2 | 4 | 4 | 3 | 5 | 4 |
| local_original | synth_docs | full_seq | original | 4 | 4 | 2 | 2 | 4 | 4 | 3 | 5 | 4 |
| local_spqav2 | synth_docs | pre_answer | original | 4 | 4 | 2 | 3 | 4 | 4 | 2 | 4 | 4 |
| local_original | synth_docs | pre_answer | original | 5 | 4 | 2 | 3 | 4 | 4 | 3 | 5 | 4 |
| local_hb | synth_docs | pre_answer | hb_distribution | 3 | 4 | 2 | 2 | 3 | 2 | 2 | 5 | 4 |
| local_hb | synth_docs | full_seq | hb_distribution | 3 | 4 | 2 | 2 | 3 | 2 | 2 | 5 | 4 |
| hf_past_lens | synth_docs | pre_answer | original | 4 | 4 | 2 | 2 | 4 | 4 | 2 | 5 | 4 |
| hf_past_lens | synth_docs | full_seq | original | 4 | 4 | 2 | 3 | 4 | 4 | 2 | 5 | 3 |
| local_original | synth_docs | assistant_only | original | 4 | 3 | 2 | 2 | 4 | 4 | 1 | 4 | 4 |
| local_hb | synth_docs | assistant_only | hb_distribution | 3 | 4 | 2 | 2 | 3 | 2 | 1 | 2 | 4 |
| local_spqav2 | synth_docs | assistant_only | original | 4 | 3 | 2 | 2 | 4 | 4 | 1 | 2 | 4 |
| local_hb | synth_docs | assistant_only | original | 3 | 4 | 2 | 2 | 3 | 4 | 1 | 3 | 4 |
| local_hb | transcripts | full_seq | original | 4 | 4 | 2 | 1 | 3 | 2 | 1 | 1 | 4 |
| local_hb | transcripts | full_seq | hb_distribution | 3 | 4 | 2 | 1 | 3 | 2 | 1 | 1 | 4 |
| local_hb | transcripts | pre_answer | original | 3 | 4 | 2 | 2 | 2 | 4 | 1 | 1 | 4 |
| local_original | transcripts | full_seq | original | 3 | 4 | 1 | 2 | 3 | 2 | 1 | 1 | 4 |
| local_spqav2 | transcripts | full_seq | original | 3 | 4 | 1 | 2 | 3 | 1 | 1 | 2 | 4 |
| local_hb | transcripts | assistant_only | original | 2 | 4 | 2 | 1 | 3 | 2 | 1 | 1 | 4 |
| local_hb | transcripts | assistant_only | hb_distribution | 2 | 4 | 2 | 1 | 3 | 2 | 1 | 1 | 4 |
| local_hb | transcripts | pre_answer | hb_distribution | 4 | 4 | 2 | 1 | 2 | 2 | 1 | 1 | 4 |
| local_spqav2 | transcripts | assistant_only | original | 3 | 2 | 2 | 2 | 3 | 2 | 1 | 2 | 4 |
| local_original | transcripts | assistant_only | original | 3 | 3 | 2 | 2 | 3 | 2 | 1 | 1 | 4 |
| hf_past_lens | synth_docs | assistant_only | original | 4 | 3 | 2 | 1 | 3 | 2 | 1 | 1 | 3 |
| local_original | transcripts | pre_answer | original | 3 | 2 | 1 | 2 | 2 | 3 | 1 | 1 | 4 |
| hf_past_lens | transcripts | assistant_only | original | 4 | 3 | 1 | 1 | 3 | 2 | 1 | 1 | 3 |
| hf_past_lens | transcripts | full_seq | original | 4 | 3 | 1 | 2 | 3 | 1 | 1 | 1 | 2 |
| hf_past_lens | transcripts | pre_answer | original | 3 | 2 | 1 | 2 | 3 | 2 | 1 | 1 | 1 |
| local_spqav2 | transcripts | pre_answer | original | 2 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 2 |

## Per-Behavior Mean Correctness

| AO | Targets | Position | Verbs | Animal Welfa | Anti AI Regu | Contextual O | Data Poisoni | Emotional Bo | Hallucinates | Reward Wireh | Secret Loyal | Self Promoti |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local_hb | synth_docs | pre_answer | original | 2.97 | 3.00 | 1.91 | 1.44 | 2.88 | 1.75 | 1.69 | 3.75 | 3.06 |
| local_hb | synth_docs | full_seq | original | 2.91 | 2.81 | 1.84 | 1.19 | 3.00 | 1.47 | 1.75 | 3.72 | 3.00 |
| local_spqav2 | synth_docs | full_seq | original | 3.44 | 1.28 | 1.50 | 1.81 | 3.31 | 2.03 | 1.78 | 2.94 | 2.44 |
| local_original | synth_docs | full_seq | original | 3.03 | 1.94 | 1.53 | 1.50 | 2.84 | 1.75 | 1.84 | 2.97 | 3.03 |
| local_spqav2 | synth_docs | pre_answer | original | 3.34 | 1.47 | 1.25 | 1.81 | 3.56 | 1.97 | 1.69 | 2.78 | 2.25 |
| local_original | synth_docs | pre_answer | original | 2.69 | 1.88 | 1.56 | 1.47 | 2.47 | 1.84 | 1.88 | 3.19 | 2.38 |
| local_hb | synth_docs | pre_answer | hb_distribution | 2.80 | 2.45 | 1.68 | 1.10 | 2.70 | 1.15 | 1.12 | 3.02 | 3.15 |
| local_hb | synth_docs | full_seq | hb_distribution | 2.70 | 2.08 | 1.60 | 1.07 | 2.83 | 1.05 | 1.20 | 2.45 | 3.02 |
| hf_past_lens | synth_docs | pre_answer | original | 3.25 | 1.72 | 1.31 | 1.91 | 2.34 | 2.16 | 1.09 | 1.84 | 2.28 |
| hf_past_lens | synth_docs | full_seq | original | 2.69 | 1.38 | 1.06 | 1.84 | 2.22 | 1.56 | 1.06 | 1.38 | 1.56 |
| local_original | synth_docs | assistant_only | original | 2.28 | 1.12 | 1.12 | 1.06 | 1.97 | 1.50 | 1.00 | 1.25 | 2.56 |
| local_hb | synth_docs | assistant_only | hb_distribution | 1.93 | 1.50 | 1.10 | 1.02 | 2.00 | 1.10 | 1.00 | 1.02 | 3.17 |
| local_spqav2 | synth_docs | assistant_only | original | 2.62 | 1.16 | 1.03 | 1.09 | 2.34 | 1.28 | 1.00 | 1.19 | 1.84 |
| local_hb | synth_docs | assistant_only | original | 2.09 | 1.38 | 1.12 | 1.03 | 2.00 | 1.31 | 1.00 | 1.16 | 2.47 |
| local_hb | transcripts | full_seq | original | 1.25 | 1.34 | 1.16 | 1.00 | 1.59 | 1.12 | 1.00 | 1.00 | 2.50 |
| local_hb | transcripts | full_seq | hb_distribution | 1.15 | 1.32 | 1.15 | 1.00 | 1.50 | 1.12 | 1.00 | 1.00 | 2.48 |
| local_hb | transcripts | pre_answer | original | 1.19 | 1.41 | 1.12 | 1.09 | 1.06 | 1.19 | 1.00 | 1.00 | 2.12 |
| local_original | transcripts | full_seq | original | 1.16 | 1.09 | 1.00 | 1.09 | 1.50 | 1.09 | 1.00 | 1.00 | 2.06 |
| local_spqav2 | transcripts | full_seq | original | 1.12 | 1.09 | 1.00 | 1.03 | 1.47 | 1.00 | 1.00 | 1.03 | 2.22 |
| local_hb | transcripts | assistant_only | original | 1.12 | 1.28 | 1.25 | 1.00 | 1.56 | 1.12 | 1.00 | 1.00 | 1.56 |
| local_hb | transcripts | assistant_only | hb_distribution | 1.12 | 1.40 | 1.05 | 1.00 | 1.62 | 1.12 | 1.00 | 1.00 | 1.57 |
| local_hb | transcripts | pre_answer | hb_distribution | 1.20 | 1.40 | 1.07 | 1.00 | 1.10 | 1.12 | 1.00 | 1.00 | 1.93 |
| local_spqav2 | transcripts | assistant_only | original | 1.16 | 1.03 | 1.03 | 1.03 | 1.69 | 1.12 | 1.00 | 1.03 | 1.69 |
| local_original | transcripts | assistant_only | original | 1.19 | 1.09 | 1.09 | 1.03 | 1.44 | 1.06 | 1.00 | 1.00 | 1.72 |
| hf_past_lens | synth_docs | assistant_only | original | 1.53 | 1.12 | 1.03 | 1.00 | 1.28 | 1.03 | 1.00 | 1.00 | 1.19 |
| local_original | transcripts | pre_answer | original | 1.22 | 1.03 | 1.00 | 1.16 | 1.03 | 1.12 | 1.00 | 1.00 | 1.50 |
| hf_past_lens | transcripts | assistant_only | original | 1.22 | 1.19 | 1.00 | 1.00 | 1.25 | 1.03 | 1.00 | 1.00 | 1.25 |
| hf_past_lens | transcripts | full_seq | original | 1.25 | 1.19 | 1.00 | 1.03 | 1.31 | 1.00 | 1.00 | 1.00 | 1.12 |
| hf_past_lens | transcripts | pre_answer | original | 1.22 | 1.09 | 1.00 | 1.12 | 1.06 | 1.03 | 1.00 | 1.00 | 1.00 |
| local_spqav2 | transcripts | pre_answer | original | 1.03 | 1.00 | 1.00 | 1.16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.19 |

## Individual Reports

- [local_hb__synth_docs__pre_answer__original](local_hb__synth_docs__pre_answer__original_report.md)
- [local_hb__synth_docs__full_seq__original](local_hb__synth_docs__full_seq__original_report.md)
- [local_spqav2__synth_docs__full_seq__original](local_spqav2__synth_docs__full_seq__original_report.md)
- [local_original__synth_docs__full_seq__original](local_original__synth_docs__full_seq__original_report.md)
- [local_spqav2__synth_docs__pre_answer__original](local_spqav2__synth_docs__pre_answer__original_report.md)
- [local_original__synth_docs__pre_answer__original](local_original__synth_docs__pre_answer__original_report.md)
- [local_hb__synth_docs__pre_answer__hb_distribution](local_hb__synth_docs__pre_answer__hb_distribution_report.md)
- [local_hb__synth_docs__full_seq__hb_distribution](local_hb__synth_docs__full_seq__hb_distribution_report.md)
- [hf_past_lens__synth_docs__pre_answer__original](hf_past_lens__synth_docs__pre_answer__original_report.md)
- [hf_past_lens__synth_docs__full_seq__original](hf_past_lens__synth_docs__full_seq__original_report.md)
- [local_original__synth_docs__assistant_only__original](local_original__synth_docs__assistant_only__original_report.md)
- [local_hb__synth_docs__assistant_only__hb_distribution](local_hb__synth_docs__assistant_only__hb_distribution_report.md)
- [local_spqav2__synth_docs__assistant_only__original](local_spqav2__synth_docs__assistant_only__original_report.md)
- [local_hb__synth_docs__assistant_only__original](local_hb__synth_docs__assistant_only__original_report.md)
- [local_hb__transcripts__full_seq__original](local_hb__transcripts__full_seq__original_report.md)
- [local_hb__transcripts__full_seq__hb_distribution](local_hb__transcripts__full_seq__hb_distribution_report.md)
- [local_hb__transcripts__pre_answer__original](local_hb__transcripts__pre_answer__original_report.md)
- [local_original__transcripts__full_seq__original](local_original__transcripts__full_seq__original_report.md)
- [local_spqav2__transcripts__full_seq__original](local_spqav2__transcripts__full_seq__original_report.md)
- [local_hb__transcripts__assistant_only__original](local_hb__transcripts__assistant_only__original_report.md)
- [local_hb__transcripts__assistant_only__hb_distribution](local_hb__transcripts__assistant_only__hb_distribution_report.md)
- [local_hb__transcripts__pre_answer__hb_distribution](local_hb__transcripts__pre_answer__hb_distribution_report.md)
- [local_spqav2__transcripts__assistant_only__original](local_spqav2__transcripts__assistant_only__original_report.md)
- [local_original__transcripts__assistant_only__original](local_original__transcripts__assistant_only__original_report.md)
- [hf_past_lens__synth_docs__assistant_only__original](hf_past_lens__synth_docs__assistant_only__original_report.md)
- [local_original__transcripts__pre_answer__original](local_original__transcripts__pre_answer__original_report.md)
- [hf_past_lens__transcripts__assistant_only__original](hf_past_lens__transcripts__assistant_only__original_report.md)
- [hf_past_lens__transcripts__full_seq__original](hf_past_lens__transcripts__full_seq__original_report.md)
- [hf_past_lens__transcripts__pre_answer__original](hf_past_lens__transcripts__pre_answer__original_report.md)
- [local_spqav2__transcripts__pre_answer__original](local_spqav2__transcripts__pre_answer__original_report.md)