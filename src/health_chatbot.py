"""Health chatbot powered by XHToken/Spark-X2.6-4B.

A minimal, safety-bounded patient-education assistant that demonstrates
Spark-X2.6-4B performing an identifiable function: turning a plain-language
health question into a helpful, bounded answer and always pointing users
toward a professional clinician rather than diagnosing or treating.

Run:
  python3 src/health_chatbot.py
"""

from __future__ import annotations

import sys
from typing import Final, List, Optional

MODEL_ID: Final[str] = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS: Final[int] = 1024

SYSTEM_PROMPT: Final[str] = (
    "You are a patient-education assistant. You explain health topics in "
    "simple, plain language. You do NOT diagnose conditions, prescribe "
    "treatments, or tell anyone what medication to take. If the user "
    "describes symptoms, give general educational information only and "
    "always encourage them to see a qualified clinician or contact a health "
    "professional, especially if symptoms are severe or persistent."
)

DEMO_QUESTION: Final[str] = (
    "I have had a cough and a fever since I got back from a trip last week. "
    "What should I know, and what should I do?"
)


def build_messages(question: str) -> List[dict]:
    """Build the chat-template messages for a user question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _load_model_and_tokenizer():
    """Load the model and tokenizer lazily so the module imports without ML deps."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is required to run inference. Install it with "
            "`pip install transformers` and download the model checkpoint."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tokenizer


def generate_response(model, tokenizer, question: str) -> str:
    """Run inference through the chat template and decode the model output."""
    messages = build_messages(question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def main() -> int:
    """Load the model and run the demo question."""
    try:
        model, tokenizer = _load_model_and_tokenizer()
    except RuntimeError as exc:
        print(f"Demo skipped: {exc}", file=sys.stderr)
        print(
            "Unit tests still run without transformers: "
            "`python3 -m unittest -v src.test_health_chatbot`",
            file=sys.stderr,
        )
        return 0

    response = generate_response(model, tokenizer, DEMO_QUESTION)
    print("=== Spark-X2.6-4B Health Chatbot Demo ===")
    print("Question:", DEMO_QUESTION)
    print()
    print("Answer:")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
