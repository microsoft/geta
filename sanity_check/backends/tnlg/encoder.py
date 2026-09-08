"""Convert text into a sequence of vectors.

Use tiktoken with tCL100kBaseBPETokenizer to convert text into a sequence of
vectors.
"""

import abc
import json
import os
from logging import getLogger

import tiktoken
import torch
from sentencepiece import SentencePieceProcessor


def load_dolly_encodings(file_path) -> tuple[dict, dict]:
    """Load encoding json file and build vocabulary dictionary from merge file.

    Args:
        file_path (str): Path to the encoding json file.

    Returns:
        tuple[dict, dict]: A tuple containing the vocabulary dictionary and
        special tokens dictionary.
    """
    # Find printable (excluding space) characters in extended ascii characters
    # [0, 31], [127, 160] are not printable ascii characters
    uchar_list = [
        uchar for uchar in range(2**8) if chr(uchar).isprintable() and chr(uchar) != " "
    ]
    byte_dict = {chr(uchar): uchar for uchar in uchar_list}

    # Append unprintable characters to the end of uchar_list
    offset = 0
    for uchar in range(2**8):
        if uchar not in uchar_list:
            uchar_list.append(uchar)
            byte_dict[chr(2**8 + offset)] = uchar
            offset += 1

    if len(uchar_list) != 2**8:
        raise ValueError("Failed to build ascii character list!")

    # Load encoding json file
    with open(file_path) as json_file:
        encoding = json.load(json_file)

    # Load merge list from json file
    bpe_merges = [tuple(merge_str.split()) for merge_str in encoding["model"]["merges"]]

    def decode_string(value: str) -> bytes:
        return bytes(byte_dict[uchar] for uchar in value)

    # Exclude certain characters
    #           \xc0 \xc1 \xf5 \xf6 \xf7 \xf8 \xf9 \xfa \xfb \xfc \xfd \xfe \xff
    exclude_list = [192, 193, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]

    # Build vocabulary dictionary from merge file, then check its consistency
    # with the one stored in json
    bpe_vocab = [bytes([uchar]) for uchar in uchar_list if uchar not in exclude_list]

    for first, second in bpe_merges:
        bpe_vocab.append(decode_string(first) + decode_string(second))

    # Load vocabulary dictionary
    vocab_dict = {decode_string(k): v for k, v in encoding["model"]["vocab"].items()}

    # Drop non-mergeable bpe tokens
    vocab_dict.pop(b"<|endoftext|>", None)
    vocab_dict.pop(b"<|padding|>", None)

    # Consistency check between merge file and vocabulary file
    if bpe_vocab != list(vocab_dict.keys()):
        raise ValueError("Merge file and vocabulary file are inconsistent")

    # Get special tokens
    special_tokens = {
        token["content"]: token["id"] for token in encoding["added_tokens"]
    }

    # Additional special tokens, see special_tokens.json
    max_id = -1
    for _, id in special_tokens.items():
        max_id = max(max_id, id)

    special_tokens["### End"] = max_id + 1
    special_tokens["### Instruction:"] = max_id + 2
    special_tokens["### Response:"] = max_id + 3

    return vocab_dict, special_tokens


class Tokenizer:
    """Tokenizer base class."""

    def __init__(self, encoding="cl100k_base"):
        """Initialize default tokenizer.

        There's a default but it must be overwritten by derived classes.
        """
        self.tokenizer = tiktoken.get_encoding(encoding)

    def tokenize(self, text: str, *args, **kwargs) -> list:
        """Convert input string in list of token values.

        Convert input string in list of token values.

        Args:
            text (str): Input string.

            args: Mandatory arguments for tokenizer. These are left unspecified
            because different tokenizers have different mandatory inputs.

            kwargs: Keyword arguments for tokenizer. Left unspecified because
            different tokenizers have different keyword inputs.

        Returns:
            list: List of integer id values for each token in the input string.
        """
        return self.tokenizer.encode(text, *args, **kwargs)

    def detokenize(self, token_ids) -> str:
        """Convert list of token values into list of strings.

        Convert list of token values into list of strings.

        Args:
            token_ids (list of ints): List of integer ids.

        Returns:
            str: String corresponding to input tokens.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)

    @property
    @abc.abstractmethod
    def eod(self) -> int:
        """Get id of terminal token.

        Get id of terminal token.

        Returns:
            int: ID of terminal token.
        """
        # Defined on derived class.
        return self.eod_id  # type: ignore

    @staticmethod
    def embed(
        input: list[int], embedding_weights: torch.Tensor, scale: float = 10.0
    ) -> torch.Tensor:
        """Embed token into vector space.

        Embed token into vector space.

        Args:
            input (List[int]): List of input tokens.

            embedding_weights (torch.Tensor): Weights to be applied to input
            tokens to generate embedded vectors.

            scale (float, optional): Scale factor to be applied to embedded
            vectors. Corresponds to the argument --mup-embedding-multiplier in
            https://agicode.visualstudio.com/TuringModelShare/_git/TNLGv4-Inference?path=test_generation.sh&version=GC459d057f7763133f17eb310e1fc010272194c517&_a=contents.
            Defaults to 10.0.

        Returns:
            torch.Tensor: Tokens embedded in a vector space.
        """
        return (
            torch.nn.functional.embedding(
                input,  # Indices of tokens.
                embedding_weights,  # Embedding matrix.
                None,  # Relevant for training only.
                None,  # Wether to cap L_p norm of embedding.
                2.0,  # P in L_p norm of capping.
                False,  # Relevant for training only.
                False,  # Relevant for training only.
            ).unsqueeze(0)
            * scale
        )  # Scale is the mup-embedding-multiplier.


class DollyTokenizer(Tokenizer):
    """Tokenizer for Dolly."""

    def __init__(self):
        """Initialize tokenizer for Dolly."""
        dolly_ranks, dolly_special = load_dolly_encodings(
            "TNLGv4/tokens/gpt_neox_12b_encodings.json"
        )
        # TODO: Figure out a way to break the regex below into multiple lines.
        self.tokenizer = tiktoken.Encoding(
            name="dolly",
            pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            mergeable_ranks=dolly_ranks,
            special_tokens=dolly_special,
        )

        self.eod_id = self.tokenizer.encode("### End", allowed_special="all")[0]


class TNLGTokenizer(Tokenizer):
    """TNLGTv4 Tokenizer class.

    Class for tokenization.
    """

    def __init__(self, encoding="cl100k_base"):
        """Create tokenizer.

        The TNLG tokenizer uses the same parameters as in the base class.
        """
        super().__init__(encoding=encoding)

        self.eod_id = self.tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
        self.fim_prefix_id = self.tokenizer.encode(
            "<|fim_prefix|>", allowed_special="all"
        )[0]
        self.fim_middle_id = self.tokenizer.encode(
            "<|fim_middle|>", allowed_special="all"
        )[0]
        self.fim_suffix_id = self.tokenizer.encode(
            "<|fim_suffix|>", allowed_special="all"
        )[0]
        self.endofprompt_id = self.tokenizer.encode(
            "<|endofprompt|>", allowed_special="all"
        )[0]
        self._special_tokens_and_ids = {
            "<|endoftext|>": self.eod_id,
            "<|fim_prefix|>": self.fim_prefix_id,
            "<|fim_middle|>": self.fim_middle_id,
            "<|fim_suffix|>": self.fim_suffix_id,
            "<|endofprompt|>": self.endofprompt_id,
        }

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.

        Getter method for vocabulary size.

        Returns:
            int: Number of symbols (words, punctuation, numbers) in vocabulary.
        """
        return self.tokenizer.n_vocab

    def vocab(self) -> dict:
        """Get the dictionary mapping symbols in the vocabulary to their ids.

        Get the dictionary mapping symbols in the vocabulary to their ids.

        Returns:
            dict: Dictionary with the format {symbol: id} where symbol is a
            string corresponding to a symbol in the vocabulary and and id is
            the id of that symbol.
        """
        base_dictionary = {
            self.tokenizer.decode([id]): id for id in range(self.eod_id - 1)
        }
        # Seems like keys from eod_id - 1 to endofprompt_id
        # are special tokens buffer that have not currently been
        # allocated in the tokenizer
        base_dictionary.update(self._special_tokens_and_ids)
        return base_dictionary

    def inv_vocab(self):
        """Get the inverse vocabulary map.

        The direct vocabulary has format {symbol: id} where symbol is a string
        corresponding to a symbol in the vocabulary and and id is the id of
        that symbol. The inverse vocabulary has format {id: symbol}.

        Returns:
            dict: Inverse vocabulary map.
        """
        return {id: key for key, id in self.vocab()}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert list of ids to list of strings.

        Convert list of ids to list of strings.

        Args:
            ids (List[int]): List of ids.

        Returns:
            List[str]: List of strings
        """
        # TODO(bapatra): This is too risky.
        return list(
            map(
                lambda x: x.decode("utf-8", errors="ignore"),
                self.tokenizer.decode_tokens_bytes(ids),
            )
        )


class LLAMATokenizer(Tokenizer):
    """LLAMA tokenizer class.

    Uses SentencePiece rather than tiktoken.
    """

    # Class-wide logger.
    logger = getLogger()

    def __init__(self, model_path: str) -> None:
        """Initialize the tokenizer.

        Requires a model path, per syntax of the Sentence Piece constructor.

        Args:
            model_path (str): path to model.
        """
        # reload tokenizer
        assert os.path.isfile(model_path), model_path

        # Linter does not understand that SentencePieceProcessor redefines
        # __init__.
        self.tokenizer = SentencePieceProcessor(model_file=model_path)  # type: ignore
        self.logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.eod_id = self.eos_id
        self.pad_id: int = self.tokenizer.pad_id()
        self.logger.info(
            "#words: "
            + f"{self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        # Linter does not understand structure of SentencePieceProcessor.
        assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()  # type: ignore

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        """Encode string into tokens for LLAMA."""
        assert type(s) is str
        t = self.tokenizer.encode(s)  # type: ignore
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]

        return t

    def decode(self, t: list[int]) -> str:
        """Decode tokens into string for LLAMA."""
        return self.tokenizer.decode(t)  # type: ignore
