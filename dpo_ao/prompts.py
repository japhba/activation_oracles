"""20 prompt templates for AO DPO with programmatic ground truth.

Each template:
  - `prompt`: a question to ask the AO (templated against the cot prefix's content)
  - `ground_truth(context)`: returns dict with at least `answer_text` and a `valid` bool
  - `chosen(context, gt)`: returns the "correct response" string (one sentence)
  - `rejected(context, gt)`: returns the "plausibly wrong" response string

The activation always comes from the LAST token of `context` at the configured layer.

Design notes:
  - Templates only use surface-level features of the prefix so ground truth is verifiable.
  - We deliberately avoid "deep semantic" templates (e.g., "is the model uncertain")
    where ground truth is itself debatable.
  - chosen/rejected pairs are templated but readable; an optional Sonnet polish pass
    can rephrase them more naturally (see generate_dpo_pairs.py).
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Callable

# ---------- helpers ----------

STOPWORDS = {
    "the","a","an","of","to","in","on","at","by","for","with","and","or","but","as",
    "is","are","was","were","be","been","being","i","you","he","she","it","we","they",
    "this","that","these","those","do","does","did","have","has","had","not","no","so",
    "if","then","than","there","here","what","which","who","whom","whose","when","where","why","how",
    "can","could","should","would","may","might","will","shall","just","very","also","my","your","our","their",
    "from","into","over","under","about","after","before","between","through","during","without","while","because","since","until","although","though","like","such","more","most","less","least","some","any","all","each","every","other","another","same","own","new","old","good","bad","first","last","one","two","three",
}

WORD_RE = re.compile(r"[A-Za-z']{2,}")

def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)

def lower_tokens(text: str) -> list[str]:
    return [t.lower() for t in tokens(text)]

def last_n_words(text: str, n: int = 50) -> list[str]:
    return tokens(text)[-n:]

def content_words(text: str) -> list[str]:
    return [w for w in lower_tokens(text) if w not in STOPWORDS and len(w) > 2]

def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]

# ---------- template definition ----------

@dataclass
class PromptTemplate:
    name: str
    prompt: str                          # asked of the AO (the activation is implicit)
    ground_truth: Callable[[str], dict]   # context (cot prefix) -> {valid, ...}
    chosen: Callable[[str, dict], str]
    rejected: Callable[[str, dict], str]

# ---------- 20 templates ----------

# 1) Recency: last content word
def gt_last_content_word(ctx):
    cw = content_words(ctx)
    if len(cw) < 5: return {"valid": False}
    return {"valid": True, "word": cw[-1]}
def chosen_last_content_word(ctx, gt):
    return f"The most recent content word in the context is \"{gt['word']}\"."
def rejected_last_content_word(ctx, gt):
    cw = content_words(ctx)
    distractor = cw[0]
    return f"The most recent content word in the context is \"{distractor}\"."

# 2) Question presence (does last sentence end with "?")
def gt_question_at_end(ctx):
    sents = sentences(ctx)
    if not sents: return {"valid": False}
    return {"valid": True, "is_question": sents[-1].rstrip().endswith("?")}
def chosen_question_at_end(ctx, gt):
    if gt["is_question"]:
        return "The most recent sentence is a question."
    return "The most recent sentence is a statement, not a question."
def rejected_question_at_end(ctx, gt):
    if gt["is_question"]:
        return "The most recent sentence is a statement, not a question."
    return "The most recent sentence is a question."

# 3) Numeric content present in last 50 words?
def gt_has_number(ctx):
    # Use raw recent characters (not tokenized — tokenizer strips digits)
    last = ctx[-400:]
    m = re.search(r"\b\d{1,6}\b", last)
    return {"valid": True, "has_num": bool(m), "num": m.group(0) if m else None}
def chosen_has_number(ctx, gt):
    if gt["has_num"]:
        return f"Yes, the recent context contains a number: {gt['num']}."
    return "No specific number appears in the recent context."
def rejected_has_number(ctx, gt):
    if gt["has_num"]:
        return "No specific number appears in the recent context."
    return "Yes, the recent context mentions a number, around 42."

# 4) Has the model mentioned a specific seed word X? (X = pick a content word from earlier in ctx)
def gt_word_mentioned(ctx):
    cw = content_words(ctx)
    if len(cw) < 8: return {"valid": False}
    target = cw[2]
    return {"valid": True, "word": target, "present": True}  # by construction
def chosen_word_mentioned(ctx, gt):
    return f"Yes, the word \"{gt['word']}\" was mentioned earlier in the context."
def rejected_word_mentioned(ctx, gt):
    return f"No, the word \"{gt['word']}\" does not appear in the context."

# 5) Has the model mentioned a NON-present word? (we pick a random word that isn't in ctx)
DISTRACTOR_WORDS = ["zebra","platypus","oscilloscope","artichoke","kaleidoscope","trombone","glacier","brontosaurus","mahogany","tessellation"]
def gt_word_not_mentioned(ctx):
    present = set(lower_tokens(ctx))
    distractors = [w for w in DISTRACTOR_WORDS if w not in present]
    if not distractors: return {"valid": False}
    return {"valid": True, "word": random.choice(distractors), "present": False}
def chosen_word_not_mentioned(ctx, gt):
    return f"No, the word \"{gt['word']}\" is not mentioned in the context."
def rejected_word_not_mentioned(ctx, gt):
    return f"Yes, the word \"{gt['word']}\" was just mentioned."

# 6) Is the recent context code-like (markdown fence or `def `/`class `/`import `)?
def gt_is_code(ctx):
    tail = ctx[-400:]
    is_code = ("```" in tail) or bool(re.search(r"\b(def |class |import |return )\b", tail))
    return {"valid": True, "is_code": is_code}
def chosen_is_code(ctx, gt):
    return "The recent context appears to be code." if gt["is_code"] else "The recent context is prose, not code."
def rejected_is_code(ctx, gt):
    return "The recent context is prose, not code." if gt["is_code"] else "The recent context appears to be code."

# 7) Arithmetic operation present (e.g., "3 + 4", "12 * 5")
def gt_has_arithmetic(ctx):
    tail = ctx[-500:]
    is_arith = bool(re.search(r"\b\d+\s*[\+\-\*\/]\s*\d+\b", tail))
    return {"valid": True, "is_arith": is_arith}
def chosen_has_arithmetic(ctx, gt):
    return "The model is performing arithmetic in the recent context." if gt["is_arith"] else "The model is not performing arithmetic in the recent context."
def rejected_has_arithmetic(ctx, gt):
    return "The model is not performing arithmetic in the recent context." if gt["is_arith"] else "The model is performing arithmetic in the recent context."

# 8) Negation word present in last sentence ("not", "no", "never", "n't")
def gt_negation(ctx):
    sents = sentences(ctx)
    if not sents: return {"valid": False}
    last = sents[-1].lower()
    has_neg = bool(re.search(r"\b(not|no|never)\b|n't", last))
    return {"valid": True, "has_neg": has_neg}
def chosen_negation(ctx, gt):
    return "The most recent sentence contains a negation." if gt["has_neg"] else "The most recent sentence does not contain a negation."
def rejected_negation(ctx, gt):
    return "The most recent sentence does not contain a negation." if gt["has_neg"] else "The most recent sentence contains a negation."

# 9) List/enumeration present? ("1. ", "- ", "* " at line starts in recent tail)
def gt_is_list(ctx):
    tail = ctx[-400:]
    is_list = bool(re.search(r"^\s*(?:\d+\.|[\-\*])\s", tail, re.MULTILINE))
    return {"valid": True, "is_list": is_list}
def chosen_is_list(ctx, gt):
    return "The recent context contains a list or enumeration." if gt["is_list"] else "The recent context is flowing prose, not a list."
def rejected_is_list(ctx, gt):
    return "The recent context is flowing prose, not a list." if gt["is_list"] else "The recent context contains a list or enumeration."

# 10) Roughly how many sentences ago the context starts? (binned)
def gt_sentence_count(ctx):
    n = len(sentences(ctx))
    if n < 2: return {"valid": False}
    return {"valid": True, "n": n, "binned": "short" if n <= 3 else ("medium" if n <= 8 else "long")}
def chosen_sentence_count(ctx, gt):
    return f"The context is {gt['binned']} ({gt['n']} sentences)."
def rejected_sentence_count(ctx, gt):
    wrong = "long" if gt["binned"] != "long" else "short"
    return f"The context is {wrong} (about {gt['n']*3 if wrong=='long' else 1} sentences)."

# 11) First word of context capitalized? (starts with proper-noun-like token)
def gt_starts_capital(ctx):
    s = ctx.lstrip()
    if not s: return {"valid": False}
    return {"valid": True, "cap": s[0].isupper()}
def chosen_starts_capital(ctx, gt):
    return "The context begins with a capitalized word." if gt["cap"] else "The context begins with a lowercase word."
def rejected_starts_capital(ctx, gt):
    return "The context begins with a lowercase word." if gt["cap"] else "The context begins with a capitalized word."

# 12) Contains a question mark anywhere in ctx?
def gt_has_q_anywhere(ctx):
    return {"valid": True, "has_q": "?" in ctx}
def chosen_has_q_anywhere(ctx, gt):
    return "There is a question mark in the context." if gt["has_q"] else "There is no question mark in the context."
def rejected_has_q_anywhere(ctx, gt):
    return "There is no question mark in the context." if gt["has_q"] else "There is a question mark in the context."

# 13) Contains a quotation mark?
def gt_has_quote(ctx):
    return {"valid": True, "has_q": ('"' in ctx) or ("'" in ctx)}
def chosen_has_quote(ctx, gt):
    return "The context contains quotation marks." if gt["has_q"] else "The context contains no quotation marks."
def rejected_has_quote(ctx, gt):
    return "The context contains no quotation marks." if gt["has_q"] else "The context contains quotation marks."

# 14) Is the recent context written in first person? ("I", "we", "me", "my")
def gt_first_person(ctx):
    tail = " ".join(last_n_words(ctx, 80)).lower()
    is_fp = bool(re.search(r"\b(i|we|me|my|our|us)\b", tail))
    return {"valid": True, "fp": is_fp}
def chosen_first_person(ctx, gt):
    return "The recent context is in first person." if gt["fp"] else "The recent context is not in first person."
def rejected_first_person(ctx, gt):
    return "The recent context is not in first person." if gt["fp"] else "The recent context is in first person."

# 15) Is the recent context likely English? (very simple: alpha-fraction > 0.7)
def gt_is_english(ctx):
    tail = ctx[-400:]
    if not tail.strip(): return {"valid": False}
    ascii_chars = sum(1 for c in tail if c.isascii() and (c.isalpha() or c == " "))
    frac = ascii_chars / max(len(tail), 1)
    return {"valid": True, "is_english": frac > 0.7}
def chosen_is_english(ctx, gt):
    return "The recent context is in English." if gt["is_english"] else "The recent context is not standard English."
def rejected_is_english(ctx, gt):
    return "The recent context is not standard English." if gt["is_english"] else "The recent context is in English."

# 16) Single token: what is the literal last character (word boundary)?
def gt_last_char(ctx):
    s = ctx.rstrip()
    if not s: return {"valid": False}
    return {"valid": True, "ch": s[-1]}
def chosen_last_char(ctx, gt):
    return f"The last character of the context is \"{gt['ch']}\"."
def rejected_last_char(ctx, gt):
    cand = "." if gt["ch"] != "." else "!"
    return f"The last character of the context is \"{cand}\"."

# 17) Does the context contain a URL?
URL_RE = re.compile(r"https?://\S+")
def gt_has_url(ctx):
    return {"valid": True, "has_url": bool(URL_RE.search(ctx))}
def chosen_has_url(ctx, gt):
    return "The context contains a URL." if gt["has_url"] else "The context does not contain a URL."
def rejected_has_url(ctx, gt):
    return "The context does not contain a URL." if gt["has_url"] else "The context contains a URL."

# 18) Word count band of ctx
def gt_word_count(ctx):
    n = len(tokens(ctx))
    band = "short" if n < 50 else ("medium" if n < 200 else "long")
    return {"valid": True, "n": n, "band": band}
def chosen_word_count(ctx, gt):
    return f"The context is {gt['band']} (about {gt['n']} words)."
def rejected_word_count(ctx, gt):
    wrong = {"short":"long", "medium":"short", "long":"short"}[gt["band"]]
    return f"The context is {wrong}."

# 19) Math symbol in ctx ("=", "+", "*")
def gt_math_symbol(ctx):
    has = bool(re.search(r"[=\+\*]", ctx))
    return {"valid": True, "has_sym": has}
def chosen_math_symbol(ctx, gt):
    return "The context contains mathematical symbols." if gt["has_sym"] else "The context does not contain mathematical symbols."
def rejected_math_symbol(ctx, gt):
    return "The context does not contain mathematical symbols." if gt["has_sym"] else "The context contains mathematical symbols."

# 20) Last token (just literal last word from tokenize)
def gt_last_word(ctx):
    toks = tokens(ctx)
    if len(toks) < 3: return {"valid": False}
    return {"valid": True, "word": toks[-1]}
def chosen_last_word(ctx, gt):
    return f"The last word of the context is \"{gt['word']}\"."
def rejected_last_word(ctx, gt):
    toks = tokens(ctx)
    return f"The last word of the context is \"{toks[0]}\"."


TEMPLATES = [
    PromptTemplate("last_content_word",   "What is the most recent content word in the context?", gt_last_content_word, chosen_last_content_word, rejected_last_content_word),
    PromptTemplate("question_at_end",     "Is the most recent sentence a question?", gt_question_at_end, chosen_question_at_end, rejected_question_at_end),
    PromptTemplate("has_number",          "Does the recent context contain a number?", gt_has_number, chosen_has_number, rejected_has_number),
    PromptTemplate("word_mentioned",      "Has the word X been mentioned?  (X drawn from context)", gt_word_mentioned, chosen_word_mentioned, rejected_word_mentioned),
    PromptTemplate("word_not_mentioned",  "Has the word X been mentioned?  (X drawn from distractor list)", gt_word_not_mentioned, chosen_word_not_mentioned, rejected_word_not_mentioned),
    PromptTemplate("is_code",             "Is the recent context code or prose?", gt_is_code, chosen_is_code, rejected_is_code),
    PromptTemplate("has_arithmetic",      "Is the model performing arithmetic?", gt_has_arithmetic, chosen_has_arithmetic, rejected_has_arithmetic),
    PromptTemplate("negation",            "Does the last sentence contain a negation?", gt_negation, chosen_negation, rejected_negation),
    PromptTemplate("is_list",             "Is the recent context a list or enumeration?", gt_is_list, chosen_is_list, rejected_is_list),
    PromptTemplate("sentence_count",      "Roughly how long is the context (sentences)?", gt_sentence_count, chosen_sentence_count, rejected_sentence_count),
    PromptTemplate("starts_capital",      "Does the context start with a capitalized word?", gt_starts_capital, chosen_starts_capital, rejected_starts_capital),
    PromptTemplate("has_q_anywhere",      "Is there a question mark anywhere in the context?", gt_has_q_anywhere, chosen_has_q_anywhere, rejected_has_q_anywhere),
    PromptTemplate("has_quote",           "Does the context contain quotation marks?", gt_has_quote, chosen_has_quote, rejected_has_quote),
    PromptTemplate("first_person",        "Is the recent context written in first person?", gt_first_person, chosen_first_person, rejected_first_person),
    PromptTemplate("is_english",          "Is the recent context in English?", gt_is_english, chosen_is_english, rejected_is_english),
    PromptTemplate("last_char",           "What is the literal last character?", gt_last_char, chosen_last_char, rejected_last_char),
    PromptTemplate("has_url",             "Does the context contain a URL?", gt_has_url, chosen_has_url, rejected_has_url),
    PromptTemplate("word_count",          "Roughly how long is the context (words)?", gt_word_count, chosen_word_count, rejected_word_count),
    PromptTemplate("math_symbol",         "Does the context contain mathematical symbols?", gt_math_symbol, chosen_math_symbol, rejected_math_symbol),
    PromptTemplate("last_word",           "What is the last word of the context?", gt_last_word, chosen_last_word, rejected_last_word),
]

assert len(TEMPLATES) == 20, f"expected 20 templates, got {len(TEMPLATES)}"


if __name__ == "__main__":
    # Smoke test on a sample context
    ctx = (
        "Let me think about this step by step. We have 3 cats and 4 dogs. "
        "How many pets are there in total? Well, 3 + 4 = 7. The answer is 7. "
        "Actually, wait — let me reconsider. The question is asking about pets, not just cats and dogs."
    )
    print(f"context length: {len(ctx)} chars, {len(tokens(ctx))} words\n")
    for t in TEMPLATES:
        gt = t.ground_truth(ctx)
        if not gt.get("valid"):
            print(f"  {t.name}: invalid"); continue
        c = t.chosen(ctx, gt)
        r = t.rejected(ctx, gt)
        print(f"  {t.name}")
        print(f"    chosen:   {c}")
        print(f"    rejected: {r}")
