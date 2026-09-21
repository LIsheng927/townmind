"""跟 transformers 版本无关的小工具。

不同版本的 transformers，tokenizer.apply_chat_template(tokenize=True, ...) 返回的类型不完全
一样——有的直接给一个 token id 列表，有的给一个字典/BatchEncoding（要从里面取 "input_ids"）。
train.py 和 evaluate.py 都要用这个功能，之前在 train.py 里踩过一次这个坑，这里统一包一层，
避免同样的错误再踩一遍。
"""


def chat_prompt_ids(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    return list(out["input_ids"] if hasattr(out, "keys") else out)
