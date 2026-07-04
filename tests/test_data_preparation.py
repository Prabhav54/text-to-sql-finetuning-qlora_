"""Basic unit tests for the SQL ChatML formatting logic (no GPU/model needed)."""

from src.data_preparation import format_example


def _cfg():
    return {
        "data": {
            "question_field": "question",
            "context_field": "context",
            "answer_field": "answer",
        }
    }


def test_format_example_contains_chatml_tags_without_tokenizer():
    example = {
        "question": "How many employees are in Sales?",
        "context": "CREATE TABLE employees (id INT, department TEXT)",
        "answer": "SELECT COUNT(*) FROM employees WHERE department = 'Sales'",
    }

    result = format_example(example, _cfg(), tokenizer=None)

    assert "<|im_start|>system" in result["text"]
    assert "<|im_start|>user" in result["text"]
    assert "<|im_start|>assistant" in result["text"]
    assert "SELECT COUNT(*)" in result["text"]
    assert "CREATE TABLE employees" in result["text"]


def test_format_example_strips_whitespace():
    example = {"question": "  hello  ", "context": "  schema  ", "answer": "  SELECT 1  "}

    result = format_example(example, _cfg(), tokenizer=None)

    assert "hello" in result["text"]
    assert "SELECT 1" in result["text"]
    assert "  hello  " not in result["text"]
