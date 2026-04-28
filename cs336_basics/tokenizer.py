from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_pair(ids: tuple[int, ...] | list[int], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return tuple(merged)


class Tokenizer:
    def __init__(self, pattern: str = GPT2_SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.byte_encoder: dict[int, int] = {i: i for i in range(256)}

        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}

        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def register_special_tokens(self, special_tokens: list[str] | dict[str, int] | None) -> None:
        if special_tokens is None:
            mapping: dict[str, int] = {}
        elif isinstance(special_tokens, dict):
            mapping = dict(special_tokens)
        elif isinstance(special_tokens, list):
            mapping = {}
            next_id = max(self.vocab, default=-1) + 1
            bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                token_id = bytes_to_id.get(token_bytes)
                if token_id is None:
                    token_id = next_id
                    next_id += 1
                    self.vocab[token_id] = token_bytes
                    bytes_to_id[token_bytes] = token_id
                mapping[token] = token_id
        else:
            raise TypeError("special_tokens must be None, list[str], or dict[str, int]")

        self.special_tokens = mapping
        self.inverse_special_tokens = {token_id: token for token, token_id in mapping.items()}

        for token, token_id in mapping.items():
            self.vocab[token_id] = token.encode("utf-8")

    def pre_tokenization(self, text: str, special_tokens: list[str] | None = None) -> dict[tuple[int, ...], int]:
        pretoken_counts: dict[tuple[int, ...], int] = {}
        pieces = [text]

        if special_tokens:
            split_pattern = "|".join(
                re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)
            )
            pieces = re.split(split_pattern, text)

        for piece in pieces:
            if not piece:
                continue
            for match in self.compiled_pattern.finditer(piece):
                token_ids = tuple(match.group(0).encode("utf-8"))
                pretoken_counts[token_ids] = pretoken_counts.get(token_ids, 0) + 1

        return pretoken_counts

    def compute_merge(
        self,
        pretoken_counts: dict[tuple[int, ...], int],
        num_merges: int,
    ) -> list[tuple[bytes, bytes]]:
        merges_in_order: list[tuple[bytes, bytes]] = []
        next_id = 256
        word_counts = dict(pretoken_counts)
        pair_counts: dict[tuple[int, int], int] = {}
        pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = {}

        for word, freq in word_counts.items():
            if len(word) < 2:
                continue
            for pair in zip(word, word[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = {word}
                else:
                    pair_to_words[pair].add(word)

        for rank in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=lambda pair: (pair_counts[pair], self.vocab[pair[0]], self.vocab[pair[1]]),
            )
            merged_id = next_id
            next_id += 1

            left_bytes = self.vocab[best_pair[0]]
            right_bytes = self.vocab[best_pair[1]]
            merged_bytes = left_bytes + right_bytes

            self.vocab[merged_id] = merged_bytes
            self.merges[best_pair] = merged_id
            self.merge_ranks[best_pair] = rank
            merges_in_order.append((left_bytes, right_bytes))

            affected_words = list(pair_to_words.get(best_pair, ()))
            merged_word_counts: dict[tuple[int, ...], int] = {}

            for word in affected_words:
                freq = word_counts.pop(word, 0)
                if freq == 0:
                    continue

                word_pairs = list(zip(word, word[1:]))
                for pair in word_pairs:
                    updated_count = pair_counts[pair] - freq
                    if updated_count == 0:
                        del pair_counts[pair]
                    else:
                        pair_counts[pair] = updated_count

                for pair in set(word_pairs):
                    words_with_pair = pair_to_words[pair]
                    words_with_pair.discard(word)
                    if not words_with_pair:
                        del pair_to_words[pair]

                merged_word = merge_pair(word, best_pair, merged_id)
                merged_word_counts[merged_word] = merged_word_counts.get(merged_word, 0) + freq

            for word, freq in merged_word_counts.items():
                word_counts[word] = word_counts.get(word, 0) + freq
                if len(word) < 2:
                    continue
                for pair in zip(word, word[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    if pair not in pair_to_words:
                        pair_to_words[pair] = {word}
                    else:
                        pair_to_words[pair].add(word)

        return merges_in_order

    def encode_pretoken(self, token_bytes: bytes) -> list[int]:
        ids = [self.byte_encoder[byte] for byte in token_bytes]

        while len(ids) >= 2:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None

            for pair in zip(ids, ids[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            ids = list(merge_pair(ids, best_pair, self.merges[best_pair]))

        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self.compiled_pattern.finditer(text):
            ids.extend(self.encode_pretoken(match.group(0).encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "all") -> list[int]:
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            if any(token in text for token in self.special_tokens):
                raise ValueError("text contains a disallowed special token")
            special = {}
        elif isinstance(allowed_special, set):
            special = {token: token_id for token, token_id in self.special_tokens.items() if token in allowed_special}
        else:
            raise ValueError(f"unsupported allowed_special={allowed_special}")

        if not special:
            return self.encode_ordinary(text)

        split_pattern = "(" + "|".join(re.escape(token) for token in sorted(special, key=len, reverse=True)) + ")"
        parts = re.split(split_pattern, text)

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            token_id = special.get(part)
            if token_id is not None:
                ids.append(token_id)
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
        allowed_special: str | set[str] = "all",
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text, allowed_special=allowed_special)

    def decode(self, ids: list[int]) -> str:
        chunks: list[bytes] = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                raise ValueError(f"invalid token id: {token_id}")
            chunks.append(token_bytes)
        return b"".join(chunks).decode("utf-8", errors="replace")

    @classmethod
    def from_vocab_merges(
        cls,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pattern: str = GPT2_SPLIT_PATTERN,
    ) -> "Tokenizer":
        tokenizer = cls(pattern=pattern)
        tokenizer.vocab = dict(vocab)
        tokenizer.byte_encoder = {
            token_bytes[0]: token_id
            for token_id, token_bytes in tokenizer.vocab.items()
            if len(token_bytes) == 1
        }

        bytes_to_id = {token_bytes: token_id for token_id, token_bytes in tokenizer.vocab.items()}
        tokenizer.merges = {}
        tokenizer.merge_ranks = {}

        for rank, (left_bytes, right_bytes) in enumerate(merges):
            merged_bytes = left_bytes + right_bytes
            if left_bytes not in bytes_to_id or right_bytes not in bytes_to_id or merged_bytes not in bytes_to_id:
                raise ValueError("vocab and merges are inconsistent")
            pair = (bytes_to_id[left_bytes], bytes_to_id[right_bytes])
            tokenizer.merges[pair] = bytes_to_id[merged_bytes]
            tokenizer.merge_ranks[pair] = rank

        special_mapping: dict[str, int] = {}
        for token in special_tokens or []:
            token_bytes = token.encode("utf-8")
            token_id = bytes_to_id.get(token_bytes)
            if token_id is None:
                raise ValueError(f"special token not in vocab: {token}")
            special_mapping[token] = token_id
        tokenizer.register_special_tokens(special_mapping)

        return tokenizer

    @classmethod
    def train_bpe(
        cls,
        input_path: str | Path,
        vocab_size: int,
        special_tokens: list[str],
        pattern: str = GPT2_SPLIT_PATTERN,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        tokenizer = cls(pattern=pattern)
        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.byte_encoder = {i: i for i in range(256)}
        tokenizer.merges = {}
        tokenizer.merge_ranks = {}
        tokenizer.special_tokens = {}
        tokenizer.inverse_special_tokens = {}

        num_merges = max(0, vocab_size - 256 - len(special_tokens or []))
        pretoken_counts = tokenizer.pre_tokenization(text, special_tokens)
        merges = tokenizer.compute_merge(pretoken_counts, num_merges)
        tokenizer.register_special_tokens(special_tokens)

        return dict(tokenizer.vocab), merges
