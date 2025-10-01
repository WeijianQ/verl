import json
from pathlib import Path

from jinja2 import Environment


def _render_template(messages):
    template_path = Path("src/qwen_2_5_memory_hf_model_py/tokenizer_config.json")
    template_str = json.loads(template_path.read_text())["chat_template"]
    env = Environment()
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=False)


def test_chat_template_skips_memory_increment_on_numbered_recall():
    system_prompt = "You are a helpful assistant for {task}."
    task = "testing"
    observations_store = [
        "first observation",
        "second observation",
        "third observation",
    ]

    messages_with_memory0 = [
        {"role": "system", "content": system_prompt.format(task=task)},
        {"role": "user", "content": [
            {"type": "memory_text", "memory_text": {"text": observations_store[0]}},
        ]},
        {"role": "assistant", "content": "dummy response"},
        {"role": "user", "content": [
            {"type": "text", "text": "dummy question"},
            {"type": "memory_text", "memory_text": {"text": observations_store[1]}},
        ]},
        {"role": "assistant", "content": "<think> I need to recall what happend in the first step.</think> <recall>1</recall> "},
        {"role": "user", "content": [
            {"type": "text", "text": f"memory 1 recalled: {observations_store[0]}"},
        ]},
        {"role": "assistant", "content": "dummy response"},
    ]

    rendered = _render_template(messages_with_memory0)
    assert rendered.count("Memory ") == 2
    assert "Memory 3" not in rendered
    assert rendered.count("<|mem_start|><|mem_pad|><|mem_end|>") == 2

    messages_with_memory_recalled_after = [
        {"role": "system", "content": system_prompt.format(task=task)},
        {"role": "user", "content": [
            {"type": "memory_text", "memory_text": {"text": observations_store[0]}},
        ]},
        {"role": "assistant", "content": "dummy response"},
        {"role": "user", "content": [
            {"type": "text", "text": "dummy question"},
            {"type": "memory_text", "memory_text": {"text": observations_store[1]}},
        ]},
        {"role": "assistant", "content": "<think> I need to recall what happend in the first step.</think> <recall>1</recall> "},
        {"role": "user", "content": [
            {"type": "text", "text": "memory 1 recalled:"},
            {"type": "memory_text", "memory_text": {"text": observations_store[0]}},
        ]},
        {"role": "assistant", "content": "dummy response"},
        {"role": "user", "content": [
            {"type": "text", "text": f"new observation: {observations_store[2]}"},
        ]},
    ]

    rendered_recall = _render_template(messages_with_memory_recalled_after)
    assert rendered_recall.count("Memory ") == 2
    assert "Memory 3" not in rendered_recall
    assert rendered_recall.count("<|mem_start|><|mem_pad|><|mem_end|>") == 3
