import nl_probes.utils.dataset_utils as dataset_utils


class StubTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        del tokenize, enable_thinking
        text = ""
        for msg in messages:
            text += f"<{msg['role']}>" + msg["content"]
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == dataset_utils.SPECIAL_TOKEN:
            return [999]
        return [ord(ch) for ch in text]


def test_multilayer_prefix_and_positions():
    tokenizer = StubTokenizer()
    prefix = dataset_utils.get_introspection_prefix(9, 6, [9, 27])
    assert prefix == "L9: ? ? ? L27: ? ? ?\n"

    datapoint = dataset_utils.create_training_datapoint(
        "test",
        "prompt",
        "answer",
        9,
        6,
        tokenizer,
        None,
        -1,
        context_input_ids=[1, 2, 3, 4, 5, 6],
        context_positions=[10, 11, 12, 10, 11, 12],
        meta_info={"layers": [9, 27]},
    )

    assert datapoint.positions == [9, 10, 11, 17, 18, 19]
