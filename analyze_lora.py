# %%

import pickle

from interp_tools.config import SelfInterpTrainingConfig

# %%
filename = "lora_eval_results_encoder_prefill.pkl"
filename = "lora_eval_results.pkl"

with open(filename, "rb") as f:
    results = pickle.load(f)

print(results["aggregated_metrics"])
# %%
config = results["config"]
print(config["prefill_original_sentences"])
training_filename = config["training_data_filename"]

training_filename = "gpt4o_contrastive_rewriting_results.pkl"
# training_filename = "gpt41_contrastive_rewriting_results.pkl"

print(training_filename)

eval_features = config["eval_features"]
print(len(eval_features))

init_eval_features = eval_features.copy()

eval_features = []

for result, sentence_data in zip(
    results["all_feature_results"], results["all_sentence_data"]
):
    if sentence_data["original_sentence"] != "":
        eval_features.append(result["feature_idx"][0])

# print(eval_features)

print(len(eval_features))
print(results.keys())
print(results["all_sentence_metrics"][0])

# %%
with open(training_filename, "rb") as f:
    training_data = pickle.load(f)

# %%

print(len(training_data))
print(training_data.keys())
print(training_data["results"][0].keys())

training_data_sentence_metrics = []
training_data_sentence_data = []

for result in training_data["results"]:
    if result["feature_idx"] in eval_features:
        training_data_sentence_metrics.append(result["sentence_metrics"][0])
        training_data_sentence_data.append(result["sentence_data"][0])

# %%
print(len(training_data_sentence_metrics))
print(len(results["all_sentence_metrics"]))
# print(training_data_sentence_metrics)
print(training_data_sentence_metrics[0])

zero_count = 0

for metric in training_data_sentence_metrics:
    if metric["original_max_activation"] == 0:
        zero_count += 1

print(f"{zero_count} / {len(training_data_sentence_metrics)}")


# %%
def aggregate_metrics(sentence_data: list[dict], sentence_metrics: list[dict]) -> dict:
    aggregated_metrics = {}
    metric_keys = sentence_metrics[0].keys()
    for key in metric_keys:
        avg_val = 0
        count = 0
        num_empty = 0
        for data, metric in zip(sentence_data, sentence_metrics):
            # if data["original_sentence"] == "":
            #     num_empty += 1
            #     continue
            avg_val += metric[key]
            count += 1
        avg_val /= count
        aggregated_metrics[key] = avg_val

        print(f"num empty: {num_empty}")
    return aggregated_metrics


aggregated_train_metrics = aggregate_metrics(
    training_data_sentence_data, training_data_sentence_metrics
)
aggregated_lora_metrics = aggregate_metrics(
    results["all_sentence_data"], results["all_sentence_metrics"]
)

# %%

print(aggregated_train_metrics)
print()
print(aggregated_lora_metrics)

# %%

for i, feature_idx in enumerate(eval_features):
    print(f"Feature {i}: {feature_idx}\n")
    print(f"Lora: {results['all_feature_results_this_eval_step'][i]['explanation']}\n")

    for result in training_data["results"]:
        if result["feature_idx"] == feature_idx:
            print(f"API: {result['explanation']}\n")
            break
    break

# %%

for i, feature_idx in enumerate(eval_features):
    print(f"Feature {i}: {feature_idx}\n")

    for result in results["all_feature_results"]:
        if result["feature_idx"] == feature_idx:
            original_lora_sentence = results["all_sentence_data"][i][
                "original_sentence"
            ]
            print(f"Lora: {original_lora_sentence}\n")
        else:
            continue

    original_api_sentence = None

    for result in training_data["results"]:
        if result["feature_idx"] == feature_idx:
            original_api_sentence = result["sentence_data"][0]["original_sentence"]
            break
    assert original_api_sentence == original_lora_sentence, (
        f"API: {original_api_sentence} != Lora: {original_lora_sentence}"
    )

    if i > 8:
        break

# %%

for i, feature_idx in enumerate(eval_features):
    print(f"Feature {i}: {feature_idx}\n")
    print(
        f"Lora: {results['all_feature_results'][i]['sentence_data']['original_sentence']}\n"
    )

    original_lora_sentence = results["all_sentence_data"][i]["original_sentence"]
    rewritten_lora_sentence = results["all_sentence_data"][i]["rewritten_sentence"]
    explanation = results["all_feature_results"][i]["explanation"]

    formatted = f"""
Positive example: {original_lora_sentence}
Negative example: {rewritten_lora_sentence}
Explanation: {explanation}
<END_OF_EXAMPLE>
"""

    print(formatted)

    # for result in training_data["results"]:
    #     if result["feature_idx"] == feature_idx:
    #         print(f"API: {result['api_prompt']}\n")
    #         print(f"API: {result['api_response']}\n")
    #         break
    break

# %%
