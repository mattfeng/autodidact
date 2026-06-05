from pathlib import Path


FORBIDDEN_EVERYWHERE = (
    "import transformers",
    "from transformers",
)

FORBIDDEN_IN_SRC = (
    "import tiktoken",
    "from tiktoken",
    "import tokenizers",
    "from tokenizers",
)


def test_source_does_not_import_forbidden_tokenizer_libraries() -> None:
    """Project source should not use forbidden tokenizer libraries."""
    for root in (Path("src"), Path("scripts")):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_EVERYWHERE:
                assert forbidden not in text, f"{path} contains {forbidden!r}"

    for path in Path("src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_IN_SRC:
            assert forbidden not in text, f"{path} contains {forbidden!r}"
