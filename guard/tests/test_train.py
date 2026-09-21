"""测试 train.py 里「怎么把一条数据变成模型输入」这部分逻辑，不需要真的加载 Qwen 模型/tokenizer——
用一个假 tokenizer 验证 prompt 部分被正确地标成 -100（不计入 loss）、答案部分保留、长度对得上。
"""
from train import LABELS, to_chat_example


class FakeTokenizer:
    """假 tokenizer：把每个字符当一个 token，只是为了验证长度/位置逻辑对不对，不需要真的下载模型。"""

    eos_token_id = -1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(len(messages[0]["content"])))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1000 + i for i in range(len(text))]}


def test_to_chat_example_masks_prompt_and_keeps_label_tokens():
    row = {"persona": "p", "context": "c", "reply": "r", "label": "ok"}
    ex = to_chat_example(row, FakeTokenizer())
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    n_answer = len("ok") + 1  # label 的 token 数 + 1 个 eos
    assert ex["labels"][:-n_answer] == [-100] * (len(ex["labels"]) - n_answer)
    assert all(t != -100 for t in ex["labels"][-n_answer:])
    assert ex["labels"][-1] == FakeTokenizer.eos_token_id


def test_to_chat_example_works_for_every_label():
    for label in LABELS:
        row = {"persona": "p", "context": "c", "reply": "r", "label": label}
        ex = to_chat_example(row, FakeTokenizer())
        assert len(ex["input_ids"]) == len(ex["labels"])
        assert ex["attention_mask"] == [1] * len(ex["input_ids"])
