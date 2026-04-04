#!/usr/bin/env python3
"""
Multi-vocabulary tokenizer script.

Supports:
  - o200k_base  (OpenAI tiktoken, used by GPT-4o / GPT-4.1)
  - HuggingFace (any model from the Hub, e.g. "meta-llama/Llama-3.1-8B")
  - SentencePiece (any .model file)

Usage examples:
  python tokenizer.py -v o200k_base        --text "Hello, world!"
  python tokenizer.py -v huggingface       -m meta-llama/Llama-3.1-8B --text "Hello, world!"
  python tokenizer.py -v sentencepiece     -m /path/to/sp.model       --text "Hello, world!"
  echo "piped text" | python tokenizer.py -v o200k_base
  python tokenizer.py -v o200k_base -f input.txt
  python tokenizer.py -v o200k_base --text "Hello" --decode 1271 2928
  python tokenizer.py -v o200k_base --text "Hello" --summary
  cat big.txt | python tokenizer.py -v o200k_base --tokens | head   # pipe-friendly
"""

import argparse
import signal
import sys
from typing import Protocol

# Exit cleanly on broken pipe (e.g. when piped to head)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# Tokenizer interface
# ---------------------------------------------------------------------------
class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def vocab_size(self) -> int: ...
    def token_strings(self, ids: list[int]) -> list[str]:
        """Return the string representation of each token id."""
        ...


# ---------------------------------------------------------------------------
# o200k_base  (tiktoken)
# ---------------------------------------------------------------------------
class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "o200k_base"):
        try:
            import tiktoken
        except ImportError:
            sys.exit("tiktoken is not installed.  Run:  pip install tiktoken")
        self.enc = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken/{encoding_name}"

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]


# ---------------------------------------------------------------------------
# HuggingFace transformers / tokenizers
# ---------------------------------------------------------------------------
class HuggingFaceTokenizer:
    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            sys.exit(
                "transformers is not installed.  Run:  pip install transformers"
            )
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.name = f"huggingface/{model_name}"

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def token_strings(self, ids: list[int]) -> list[str]:
        return self.tok.convert_ids_to_tokens(ids)


# ---------------------------------------------------------------------------
# SentencePiece
# ---------------------------------------------------------------------------
class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            sys.exit(
                "sentencepiece is not installed.  Run:  pip install sentencepiece"
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.name = f"sentencepiece/{model_path}"

    def encode(self, text: str) -> list[int]:
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds(ids)

    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.sp.IdToPiece(i) for i in ids]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
VOCAB_ALIASES = {
    "o200k_base": "o200k_base",
    "o200k": "o200k_base",
    "cl100k_base": "cl100k_base",
    "cl100k": "cl100k_base",
    "p50k_base": "p50k_base",
    "p50k": "p50k_base",
}


def make_tokenizer(vocab: str, model: str | None) -> Tokenizer:
    v = vocab.lower()
    if v in VOCAB_ALIASES:
        return TiktokenTokenizer(VOCAB_ALIASES[v])
    if v in ("huggingface", "hf"):
        if not model:
            sys.exit("HuggingFace vocab requires --model <repo/name>")
        return HuggingFaceTokenizer(model)
    if v in ("sentencepiece", "sp"):
        if not model:
            sys.exit("SentencePiece vocab requires --model <path/to/sp.model>")
        return SentencePieceTokenizer(model)
    sys.exit(
        f"Unknown vocab '{vocab}'.  Choose from: "
        f"{', '.join(sorted(set(VOCAB_ALIASES) | {'huggingface', 'sentencepiece'}))}"
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_encode_result(
    tok: Tokenizer, text: str, *, verbose: bool, tokens: bool, ids_only: bool
) -> None:
    ids = tok.encode(text)

    # Pipe-friendly: one token per line
    if tokens:
        if ids_only:
            for tid in ids:
                sys.stdout.write(f"{tid}\n")
        else:
            strings = tok.token_strings(ids)
            for tid, s in zip(ids, strings):
                sys.stdout.write(f"{tid}\t{s!r}\n")
        return

    # Pipe-friendly: bare ids, one per line
    if ids_only:
        for tid in ids:
            sys.stdout.write(f"{tid}\n")
        return

    # Human-readable summary
    print(f"Tokenizer : {tok.name}")
    print(f"Vocab size: {tok.vocab_size():,}")
    print(f"Text len  : {len(text):,} chars")
    print(f"Token cnt : {len(ids):,}")
    if len(ids) <= 20:
        print(f"Token ids : {ids}")
    else:
        print(f"Token ids : [{ids[0]}, {ids[1]}, {ids[2]}, ..., {ids[-1]}] (use --tokens to see all)")
    if verbose:
        strings = tok.token_strings(ids)
        print("Tokens    :")
        for tid, s in zip(ids, strings):
            print(f"  {tid:>8d}  {s!r}")


def print_decode_result(tok: Tokenizer, ids: list[int]) -> None:
    text = tok.decode(ids)
    print(f"Tokenizer : {tok.name}")
    print(f"Input ids : {ids}")
    print(f"Decoded   : {text}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize text with multiple vocabulary backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--vocab",
        required=True,
        help="Vocabulary / backend: o200k_base, cl100k_base, huggingface, sentencepiece",
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name or path (required for huggingface / sentencepiece)",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Text to tokenize")
    input_group.add_argument("-f", "--file", help="File to tokenize")

    parser.add_argument(
        "--decode",
        nargs="+",
        type=int,
        metavar="ID",
        help="Decode token ids back to text (instead of encoding)",
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Output one token per line (id<TAB>string), pipe-friendly",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Output only token ids, one per line",
    )
    parser.add_argument(
        "-summary", "--summary",
        action="store_true",
        help="Print summary (token count, char count)",
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Show per-token string breakdown",
    )

    args = parser.parse_args()
    tok = make_tokenizer(args.vocab, args.model)

    # Decode mode
    if args.decode:
        print_decode_result(tok, args.decode)
        return

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file) as fh:
            text = fh.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide text via -t, -f, or stdin")

    if args.summary:
        ids = tok.encode(text)
        print(f"Tokens: {len(ids)}")
        print(f"Chars:  {len(text)}")
        return

    print_encode_result(
        tok, text, verbose=args.verbose, tokens=args.tokens, ids_only=args.ids_only
    )


if __name__ == "__main__":
    main()
