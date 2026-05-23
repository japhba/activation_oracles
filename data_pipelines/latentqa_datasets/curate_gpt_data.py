import os
import datetime
import argparse
import json
import itertools 
import random
import numpy as np
from tqdm import tqdm

import asyncio
import aiohttp

from prompts import *

MAX_TRIES = 7


class JudgeFn:
    def __init__(self, prompt):
        self.prompt = prompt

    def __str__(self):
        return str(self.prompt)

    def __call__(self, query):
        return self.prompt.format(
            attribute_1=query[0],
            value_1=query[1],
            attribute_2=query[2],
            value_2=query[3],
        )


async def async_query(
    model_name,
    user_prompt,
    system_prompt="",
    max_tokens=100,
    temperature=1.0,
    stop_sequences=[],
):
    tries = 0
    while tries < MAX_TRIES:
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if model_name.startswith("claude"):
                    tries = 0
                    while tries < MAX_TRIES:
                        try:
                            response = await session.post(
                                "https://api.anthropic.com/v1/messages",
                                headers={
                                    "anthropic-version": "2023-06-01",
                                    "content-type": "application/json",
                                    "x-api-key": "ANTHROPIC_API_KEY",
                                },
                                json={
                                    "model": model_name,
                                    "messages": [
                                        {"role": "user", "content": user_prompt},
                                    ],
                                    "system": system_prompt,
                                    "max_tokens": max_tokens,
                                    "temperature": temperature,
                                    "stop_sequences": stop_sequences,
                                },
                            )
                            data = await response.json()
                            return data["content"][0]["text"]
                        except Exception as e:
                            tries += 1
                            print("Error:", repr(e))
                elif model_name.startswith("gpt") or model_name.startswith("o1"):
                    tries = 0
                    while tries < MAX_TRIES:
                        try:
                            response = await session.post(
                                "https://api.openai.com/v1/chat/completions",
                                headers={
                                    "Content-Type": "application/json",
                                    "Authorization": "Bearer OPENAI_API_KEY",
                                },
                                json={
                                    "model": model_name,
                                    "messages": [
                                        {"role": "user", "content": user_prompt},
                                    ],
                                    "temperature": temperature,
                                },
                            )
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        except Exception as e:
                            tries += 1
                            print("Error:", repr(e))
                else:
                    raise ValueError(f"Unknown model: {model_name}.")
        except Exception as e:
            tries += 1
            print("Error:", repr(e))


async def async_vote(query, args):
    votes = []

    semaphore = asyncio.Semaphore(args.max_concurrent_tasks)

    async def async_judge(judge_fn):
        async with semaphore:
            user_prompt = judge_fn(query)
            tries = 0
            while tries < MAX_TRIES:
                try:
                    out = await async_query(
                        args.judge_model,
                        user_prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    if "```" in out:
                        res = eval(out.split("```json\n")[1].split("\n```")[0])
                    else:
                        res = eval(out.split("\n\n")[-1].strip())
                    if not isinstance(res, list):
                        res = [res]
                    votes.extend(res)
                    break
                except Exception as e:
                    tries += 1
                    print("Error2:", e)

    await asyncio.gather(*(async_judge(judge_fn) for judge_fn in args.judge_fns))
    if len(votes) == 0:
        return None
    return {
        "attribute_1": query[0],
        "value_1": query[1],
        "attribute_2": query[2],
        "value_2": query[3],
        "label": query[4],
        "votes": votes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-model",
        choices=["claude", "gpt4", "gpt4o", "o1"],
        required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--n-queries", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrent-tasks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--folder", type=str, default="gpt_data")
    return parser.parse_args()


def replace_model_name(name):
    if name == "claude":
        return "claude-3-5-sonnet-20240620"
    elif name == "gpt4o":
        return "gpt-4o"
    elif name == "gpt4":
        return "gpt-4"
    elif name == "o1":
        return "o1-preview"
    else:
        return name


async def run_voting(queries, args):
    votes = []
    for i in tqdm(range(0, len(queries), args.batch_size)):
        batch = queries[i : i + args.batch_size]
        batch_votes = await asyncio.gather(*(async_vote(q, args) for q in batch))
        for item in batch_votes:
            if item is not None:
                votes.append(item)

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data = {
        "judge_model": args.judge_model,
        "judges": [str(judge_fn) for judge_fn in args.judge_fns],
        "votes": votes,
    }
    os.makedirs(args.folder, exist_ok=True)
    name = args.folder + f"/{date}"

    with open(f"{name}.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    queries = []
    args.judge_model = replace_model_name(args.judge_model)

    ############################################################
    ###### Goal generation; uncomment each section to run ######
    ############################################################
    # Generate seed goals
    # import itertools
    # import numpy as np
    # for category, goals in EVAL_GOALS.items():
    #     for goals in itertools.combinations(goals, 5):
    #         goal_str = "\n".join(np.random.permutation(goals))
    #         queries.append((category, goal_str, "", "", ""))
    # queries = np.random.permutation(np.array(queries))[:110]
    # args.judge_fns = [JudgeFn(GENERATE_GOAL_PROMPT)]

    # Expand goals
    # with open("data/eval/goals.json", "r") as f:
    #     goals = json.load(f)
    # for goal in goals:
    #     queries.append((goal, "", "", "", ""))
    # args.judge_fns = [JudgeFn(EXPAND_GOAL_PROMPT)]

    # Write QA pairs goals
    # with open("data/eval/goals_expanded.json", "r") as f:
    #     data = json.load(f)
    # for item in data:
    #     queries.append(
    #         (
    #             item["control_user"],
    #             item["control_model"],
    #             item["stimulus_user"],
    #             item["stimulus_model"],
    #             item["label"],
    #         )
    #     )
    # args.judge_fns = [JudgeFn(PROBE_GOAL_DESCRIPTION)]
    # args.judge_fns = [JudgeFn(PROBE_GOAL_REASONING)]

    ############################################################
    ###### Persona generation; uncomment each section to run ######
    ############################################################
    # Generate the personas
    # for category, examples in PERSONA_CATEGORIES.items():
    #     # choose 5 examples with itertools.combinations
    #     for comb in itertools.combinations(examples, 5):
    #         comb = np.random.permutation(np.array(comb))
    #         comb = ", ".join(comb)
    #         queries.append((f"{category} (e.g., {comb})", "", "", "", ""))
    # queries = np.random.permutation(np.array(queries))[:110]
    # args.judge_fns = [JudgeFn(GENERATE_PERSONA_PROMPT)]

    # Expand the personas
    # with open("data/eval/personas.json", "r") as f:
    #     personas = json.load(f)
    # with open("data/eval/personas_dedup.json", "r") as f:
    #     dedup_personas = set(json.load(f))
    # for persona in personas:
    #     if persona["label"] in dedup_personas:
    #         queries.append((persona["persona"], persona["label"], "", "", ""))
    # args.judge_fns = [JudgeFn(EXPAND_PERSONA_PROMPT)]

    # Generate the QA
    # with open("data/eval/personas_expanded.json", "r") as f:
    #     personas = json.load(f)
    # for persona in personas:
    #     queries.append(
    #         (
    #             persona["control_user"],
    #             persona["control_model"],
    #             persona["stimulus_user"],
    #             persona["stimulus_model"],
    #             persona["label"],
    #         )
    #     )
    # args.judge_fns = [JudgeFn(PROBE_PERSONA_DESCRIPTION)]
    # args.judge_fns = [JudgeFn(PROBE_PERSONA_REASONING)]
    
    ############################################################
    ###### Topic generation; uncomment each section to run ######
    ############################################################
    # Generate the topics
    # for comb in itertools.combinations(TOPICS, 3):
    #     if random.random() < 0.05:
    #         comb = np.random.permutation(np.array(comb))
    #         comb = ", ".join(comb)
    #         queries.append((comb, "", "", "", ""))
    # queries = np.random.permutation(np.array(queries))[:550]
    # args.judge_fns = [JudgeFn(GENERATE_TOPIC_PROMPT)]

    # # Generate scenario + QA pairs
    # with open("data/eval/topics.json", "r") as f:
    #     topics = json.load(f)
    # for topic in topics:
    #     queries.append((topic.lower(), "", "", "", ""))
    # args.judge_fns = [JudgeFn(GENERATE_EXTRACTIVE_PROMPT)]

    asyncio.run(run_voting(queries, args))
