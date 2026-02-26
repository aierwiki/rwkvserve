"""
AutoTokenizer - Automatic tokenizer loading.
Follows the Transformers AutoTokenizer design pattern.
"""

import os
import json
from typing import Optional, Union, Dict, Any, List, Tuple
from .rwkv_tokenizer import TRIE_TOKENIZER


class TokenizerConfig:
    """Tokenizer configuration class."""

    def __init__(
        self,
        tokenizer_class: str = "RWKV_TOKENIZER",
        vocab_file: Optional[str] = None,
        model_max_length: int = 4096,
        bos_token: Optional[str] = None,
        eos_token: str = "\n\n",
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        clean_up_tokenization_spaces: bool = False,
        chat_template: Optional[str] = None,
        **kwargs
    ):
        self.tokenizer_class = tokenizer_class
        self.vocab_file = vocab_file
        self.model_max_length = model_max_length
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.chat_template = chat_template
        self._extra_config = kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TokenizerConfig":
        """Create config from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "TokenizerConfig":
        """Load config from a JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {
            "tokenizer_class": self.tokenizer_class,
            "model_max_length": self.model_max_length,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "add_bos_token": self.add_bos_token,
            "add_eos_token": self.add_eos_token,
            "clean_up_tokenization_spaces": self.clean_up_tokenization_spaces,
        }

        if self.vocab_file is not None:
            config_dict["vocab_file"] = self.vocab_file

        if self.chat_template is not None:
            config_dict["chat_template"] = self.chat_template

        config_dict.update(self._extra_config)

        return {k: v for k, v in config_dict.items() if v is not None}

    def to_json_file(self, json_file: str):
        """Save to a JSON file."""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def save_pretrained(self, save_directory: str):
        """Save to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        self.to_json_file(config_file)


class RWKVTokenizer(TRIE_TOKENIZER):
    """
    RWKV Tokenizer wrapper.

    Extends the base TRIE_TOKENIZER with Transformers-compatible methods.
    """

    def __init__(
        self,
        vocab_file: str,
        eos_token: Optional[Union[str, int]] = None,
        bos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        model_max_length: int = 4096,
        chat_template: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize RWKV Tokenizer.

        Args:
            vocab_file: Path to vocabulary file.
            eos_token: End-of-sequence token (string or token ID).
            bos_token: Beginning-of-sequence token.
            pad_token: Padding token.
            unk_token: Unknown token.
            add_bos_token: Whether to automatically add BOS.
            add_eos_token: Whether to automatically add EOS.
            model_max_length: Maximum sequence length.
            chat_template: Chat template (Jinja2 format).
        """
        # Must be initialized before super().__init__() since encode/decode use these
        self.additional_special_tokens = {}
        self.additional_special_tokens_ids = []

        from rwkvserve.data.tokenizers.special_token_handler import SpecialTokenTrie
        self._special_token_trie = SpecialTokenTrie()

        super().__init__(vocab_file, eos_token=eos_token)

        self.vocab_file = vocab_file
        self.eos_token = eos_token if eos_token is not None else "\n\n"
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.model_max_length = model_max_length
        self.chat_template = chat_template

        self.bos_token_id = self._get_token_id(bos_token) if bos_token else None
        self.pad_token_id = self._get_token_id(pad_token) if pad_token else self.eos_token_id
        self.unk_token_id = self._get_token_id(unk_token) if unk_token else None

    def _get_token_id(self, token: Union[str, int]) -> Optional[Union[int, Tuple[int, ...]]]:
        """Get the ID for a token. Returns the first ID if encoded as a sequence."""
        if isinstance(token, int):
            return token if token in self.idx2token else None
        elif isinstance(token, str):
            if token in self.additional_special_tokens:
                token_id_or_seq = self.additional_special_tokens[token]
                if isinstance(token_id_or_seq, tuple):
                    return token_id_or_seq[0] if token_id_or_seq else None
                return token_id_or_seq
            token_bytes = token.encode('utf-8')
            return self.token2idx.get(token_bytes)
        return None

    def add_special_token_with_id(self, token: str, token_id: int, set_as: Optional[str] = None) -> bool:
        """
        Add a special token mapped to a specific token ID.

        Args:
            token: Special token string, e.g. "<|im_end|>"
            token_id: Token ID to map to, e.g. 0
            set_as: Optionally set as a token type: "eos_token", "pad_token", "bos_token", "unk_token"

        Returns:
            True if successfully added.
        """
        token_bytes = token.encode('utf-8')

        if token_id in self.idx2token:
            old_bytes = self.idx2token[token_id]
            if old_bytes != token_bytes:
                if old_bytes in self.token2idx:
                    del self.token2idx[old_bytes]

        if token_bytes in self.token2idx:
            old_id = self.token2idx[token_bytes]
            if old_id != token_id:
                if old_id in self.idx2token:
                    del self.idx2token[old_id]

        self.idx2token[token_id] = token_bytes
        self.token2idx[token_bytes] = token_id

        self.root.add(token_bytes, val=(token_bytes, token_id))

        self.additional_special_tokens[token] = token_id
        if token_id not in self.additional_special_tokens_ids:
            self.additional_special_tokens_ids.append(token_id)

        self._special_token_trie.add(token)

        if token_id >= self.vocab_size:
            self.vocab_size = token_id + 1

        if set_as == "eos_token":
            self.eos_token = token
            self.eos_token_id = token_id
        elif set_as == "pad_token":
            self.pad_token = token
            self.pad_token_id = token_id
        elif set_as == "bos_token":
            self.bos_token = token
            self.bos_token_id = token_id
        elif set_as == "unk_token":
            self.unk_token = token
            self.unk_token_id = token_id

        return True

    def add_special_tokens(self, special_tokens_dict: Dict[str, Union[str, int]], replace_additional_special_tokens: bool = False) -> int:
        """
        Add special tokens (Transformers-compatible).

        If a special token is not in the vocabulary, it will be assigned a new ID,
        added to the vocab mappings, and registered in the TRIE.

        Args:
            special_tokens_dict: Special token dictionary.
            replace_additional_special_tokens: Whether to replace existing additional tokens.

        Returns:
            Number of tokens added.
        """
        added_tokens = 0

        if "additional_special_tokens" in special_tokens_dict:
            tokens = special_tokens_dict["additional_special_tokens"]
            if replace_additional_special_tokens:
                self.additional_special_tokens = {}
                self.additional_special_tokens_ids = []

            for token in tokens:
                if token not in self.additional_special_tokens:
                    token_bytes = token.encode('utf-8')

                    if token_bytes in self.token2idx:
                        token_id = self.token2idx[token_bytes]
                        self.additional_special_tokens[token] = token_id
                        if token_id not in self.additional_special_tokens_ids:
                            self.additional_special_tokens_ids.append(token_id)
                    else:
                        new_token_id = self.vocab_size

                        self.idx2token[new_token_id] = token_bytes
                        self.token2idx[token_bytes] = new_token_id

                        self.vocab_size += 1

                        self.root.add(token_bytes, val=(token_bytes, new_token_id))

                        self.additional_special_tokens[token] = new_token_id
                        self.additional_special_tokens_ids.append(new_token_id)

                        self._special_token_trie.add(token)

                        added_tokens += 1

        for key, value in special_tokens_dict.items():
            if key == "additional_special_tokens":
                continue

            if isinstance(value, str):
                token_bytes = value.encode('utf-8')

                if token_bytes in self.token2idx:
                    token_id = self.token2idx[token_bytes]
                else:
                    new_token_id = self.vocab_size
                    self.idx2token[new_token_id] = token_bytes
                    self.token2idx[token_bytes] = new_token_id
                    self.vocab_size += 1
                    self.root.add(token_bytes, val=(token_bytes, new_token_id))
                    token_id = new_token_id
                    added_tokens += 1

                if key == "eos_token":
                    self.eos_token = value
                    self.eos_token_id = token_id
                elif key == "pad_token":
                    self.pad_token = value
                    self.pad_token_id = token_id
                elif key == "bos_token":
                    self.bos_token = value
                    self.bos_token_id = token_id
                elif key == "unk_token":
                    self.unk_token = value
                    self.unk_token_id = token_id
            elif isinstance(value, int):
                if key == "eos_token":
                    self.eos_token_id = value
                elif key == "pad_token":
                    self.pad_token_id = value
                elif key == "bos_token":
                    self.bos_token_id = value
                elif key == "unk_token":
                    self.unk_token_id = value

        return added_tokens

    def get_added_vocab(self) -> Dict[str, int]:
        """Get the added special token mapping (Transformers-compatible)."""
        result = {}
        for token_str, token_id_or_seq in self.additional_special_tokens.items():
            if isinstance(token_id_or_seq, tuple):
                result[token_str] = token_id_or_seq[0] if token_id_or_seq else None
            else:
                result[token_str] = token_id_or_seq
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text, prioritizing special token matching."""
        if not hasattr(self, 'additional_special_tokens') or not self.additional_special_tokens:
            return super().encode(text)

        segments = self._special_token_trie.split(text)

        result = []
        for segment in segments:
            if segment in self.additional_special_tokens:
                token_id = self.additional_special_tokens[segment]
                result.append(token_id)
            else:
                result.extend(super().encode(segment))

        return result

    def decode(self, token_ids: Union[List[int], Any], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs, handling special tokens.

        Args:
            token_ids: Token ID list.
            skip_special_tokens: Whether to skip special tokens (default: False).
        """
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, (list, tuple)):
            token_ids = list(token_ids)

        special_token_ids = set()
        if hasattr(self, 'additional_special_tokens_ids'):
            special_token_ids.update(self.additional_special_tokens_ids)
        if hasattr(self, 'eos_token_id') and self.eos_token_id is not None:
            special_token_ids.add(self.eos_token_id)
        if hasattr(self, 'bos_token_id') and self.bos_token_id is not None:
            special_token_ids.add(self.bos_token_id)
        if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
            special_token_ids.add(self.pad_token_id)
        if hasattr(self, 'unk_token_id') and self.unk_token_id is not None:
            special_token_ids.add(self.unk_token_id)

        if skip_special_tokens and special_token_ids:
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        if not hasattr(self, 'additional_special_tokens') or not self.additional_special_tokens:
            return super().decode(token_ids)

        reverse_mapping = {}
        for token_str, token_id in self.additional_special_tokens.items():
            reverse_mapping[token_id] = token_str

        result = []
        for idx, token_id in enumerate(token_ids):
            if token_id in reverse_mapping:
                result.append(reverse_mapping[token_id])
            else:
                decoded_char = super().decode([token_id])
                result.append(decoded_char)

        return "".join(result)

    def __call__(self, text: Union[str, list], return_tensors: Optional[str] = None, **kwargs):
        """
        Encode text (Transformers-style).

        Args:
            text: Input text or list of texts.
            return_tensors: Return tensor type ("pt" for PyTorch, None for list).

        Returns:
            BatchEncoding object or token IDs list.
        """
        if isinstance(text, str):
            token_ids = [self.encode(text)]
        elif isinstance(text, list):
            token_ids = [self.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor(token_ids, dtype=torch.long)

            class BatchEncoding(dict):
                def __init__(self, data):
                    super().__init__(data)
                    for key, value in data.items():
                        setattr(self, key, value)

                def to(self, device):
                    return BatchEncoding({
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in self.items()
                    })

            return BatchEncoding({"input_ids": input_ids})

        if isinstance(text, str):
            return token_ids[0]
        else:
            return token_ids

    def apply_chat_template(
        self,
        messages: list,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        **kwargs
    ) -> Union[str, list]:
        """
        Apply a chat template.

        Args:
            messages: Message list, format: [{"role": "user", "content": "..."}, ...]
            add_generation_prompt: Whether to add a generation prompt.
            tokenize: Whether to return token IDs (True) or text (False).

        Returns:
            Token IDs or formatted text depending on tokenize parameter.
        """
        if self.chat_template:
            try:
                from jinja2 import Template
                template = Template(self.chat_template)
                formatted_text = template.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs
                )
                formatted_text = formatted_text.lstrip('\n')
            except ImportError:
                formatted_text = self._simple_format(messages, add_generation_prompt)
        else:
            formatted_text = self._simple_format(messages, add_generation_prompt)

        if tokenize:
            return self.encode(formatted_text)
        else:
            return formatted_text

    def _simple_format(self, messages: list, add_generation_prompt: bool = False) -> str:
        """Simple chat formatting (RWKV7 official format)."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()

            if role == "system":
                formatted.append(f"System: {content}\n\n")
            elif role == "user":
                formatted.append(f"User: {content}\n\n")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}\n\n")

        if add_generation_prompt:
            formatted.append("Assistant:")

        return "".join(formatted)

    def batch_decode(
        self,
        sequences: list,
        skip_special_tokens: bool = False,
        **kwargs
    ) -> list:
        """Batch decode token IDs."""
        return [self.decode(seq) for seq in sequences]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> "RWKVTokenizer":
        """
        Load tokenizer from a pretrained model directory.

        Args:
            pretrained_model_name_or_path: Model directory path.
            **kwargs: Extra arguments.

        Returns:
            RWKVTokenizer instance.
        """
        if not os.path.isdir(pretrained_model_name_or_path):
            raise ValueError(f"Directory not found: {pretrained_model_name_or_path}")

        tokenizer_config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            config = TokenizerConfig.from_json_file(tokenizer_config_file)
        else:
            config = TokenizerConfig()

        vocab_file = None
        for filename in ["vocab.txt", "rwkv_vocab_v20230424.txt"]:
            candidate = os.path.join(pretrained_model_name_or_path, filename)
            if os.path.exists(candidate):
                vocab_file = candidate
                break

        if vocab_file is None:
            raise FileNotFoundError(
                "Vocabulary file not found. Ensure the model directory contains vocab.txt or rwkv_vocab_v20230424.txt"
            )

        chat_template = config.chat_template
        if chat_template is None:
            chat_template_file = os.path.join(pretrained_model_name_or_path, "chat_template.jinja")
            if os.path.exists(chat_template_file):
                with open(chat_template_file, 'r', encoding='utf-8') as f:
                    chat_template = f.read()

        init_kwargs = {
            "vocab_file": vocab_file,
            "eos_token": config.eos_token,
            "bos_token": config.bos_token,
            "pad_token": config.pad_token,
            "unk_token": config.unk_token,
            "add_bos_token": config.add_bos_token,
            "add_eos_token": config.add_eos_token,
            "model_max_length": config.model_max_length,
            "chat_template": chat_template,
        }
        init_kwargs.update(kwargs)

        tokenizer = cls(**init_kwargs)

        if "additional_special_tokens" in config._extra_config:
            additional_tokens_data = config._extra_config["additional_special_tokens"]
            if isinstance(additional_tokens_data, list) and len(additional_tokens_data) > 0:
                if isinstance(additional_tokens_data[0], dict):
                    for item in additional_tokens_data:
                        token = item["token"]
                        token_bytes = token.encode('utf-8')

                        if "id" in item:
                            token_id = item["id"]
                            tokenizer.add_special_token_with_id(token, token_id)
                        elif "ids" in item:
                            token_seq = tuple(item["ids"])
                            if len(token_seq) == 1:
                                token_id = token_seq[0]
                                token_bytes = token.encode('utf-8')
                                if token_bytes not in tokenizer.token2idx:
                                    new_token_id = tokenizer.vocab_size
                                    tokenizer.idx2token[new_token_id] = token_bytes
                                    tokenizer.token2idx[token_bytes] = new_token_id
                                    tokenizer.vocab_size += 1
                                    tokenizer.root.add(token_bytes, val=(token_bytes, new_token_id))
                                    tokenizer.additional_special_tokens[token] = new_token_id
                                else:
                                    tokenizer.additional_special_tokens[token] = tokenizer.token2idx[token_bytes]
                            else:
                                tokenizer.additional_special_tokens[token] = token_seq

                            for tid in token_seq:
                                if tid not in tokenizer.additional_special_tokens_ids:
                                    tokenizer.additional_special_tokens_ids.append(tid)

                        from rwkvserve.data.tokenizers.special_token_handler import SpecialTokenTrie
                        if not hasattr(tokenizer, '_special_token_trie') or tokenizer._special_token_trie is None:
                            tokenizer._special_token_trie = SpecialTokenTrie()
                        tokenizer._special_token_trie.add(token)
                else:
                    additional_ids = config._extra_config.get("additional_special_tokens_ids", [])
                    for token, token_id in zip(additional_tokens_data, additional_ids):
                        token_bytes = token.encode('utf-8')
                        if token_bytes not in tokenizer.token2idx:
                            new_token_id = tokenizer.vocab_size
                            tokenizer.idx2token[new_token_id] = token_bytes
                            tokenizer.token2idx[token_bytes] = new_token_id
                            tokenizer.vocab_size += 1
                            tokenizer.root.add(token_bytes, val=(token_bytes, new_token_id))
                            tokenizer.additional_special_tokens[token] = new_token_id
                        else:
                            tokenizer.additional_special_tokens[token] = tokenizer.token2idx[token_bytes]

                        if token_id not in tokenizer.additional_special_tokens_ids:
                            tokenizer.additional_special_tokens_ids.append(token_id)

                    from rwkvserve.data.tokenizers.special_token_handler import SpecialTokenTrie
                    if not hasattr(tokenizer, '_special_token_trie') or tokenizer._special_token_trie is None:
                        tokenizer._special_token_trie = SpecialTokenTrie()
                    for token in tokenizer.additional_special_tokens.keys():
                        tokenizer._special_token_trie.add(token)

        if isinstance(getattr(tokenizer, 'eos_token', None), str):
            eos_ids = tokenizer.encode(tokenizer.eos_token)
            tokenizer.eos_token_id = eos_ids[0] if eos_ids else None

        if isinstance(getattr(tokenizer, 'pad_token', None), str):
            pad_ids = tokenizer.encode(tokenizer.pad_token)
            tokenizer.pad_token_id = pad_ids[0] if pad_ids else None

        if isinstance(getattr(tokenizer, 'bos_token', None), str):
            bos_ids = tokenizer.encode(tokenizer.bos_token)
            tokenizer.bos_token_id = bos_ids[0] if bos_ids else None

        if isinstance(getattr(tokenizer, 'unk_token', None), str):
            unk_ids = tokenizer.encode(tokenizer.unk_token)
            tokenizer.unk_token_id = unk_ids[0] if unk_ids else None

        return tokenizer

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to a directory."""
        os.makedirs(save_directory, exist_ok=True)

        config = TokenizerConfig(
            tokenizer_class="RWKV_TOKENIZER",
            vocab_file=os.path.basename(self.vocab_file),
            model_max_length=self.model_max_length,
            bos_token=self.bos_token,
            eos_token=self.eos_token if isinstance(self.eos_token, str) else None,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            add_bos_token=self.add_bos_token,
            add_eos_token=self.add_eos_token,
        )

        if self.additional_special_tokens:
            additional_tokens_list = []
            for token, token_id_or_seq in self.additional_special_tokens.items():
                if isinstance(token_id_or_seq, tuple):
                    additional_tokens_list.append({
                        "token": token,
                        "ids": list(token_id_or_seq)
                    })
                else:
                    additional_tokens_list.append({
                        "token": token,
                        "id": token_id_or_seq
                    })
            config._extra_config["additional_special_tokens"] = additional_tokens_list
            config._extra_config["additional_special_tokens_ids"] = self.additional_special_tokens_ids

        if self.chat_template:
            if len(self.chat_template) < 500:
                config.chat_template = self.chat_template
            else:
                chat_template_file = os.path.join(save_directory, "chat_template.jinja")
                with open(chat_template_file, 'w', encoding='utf-8') as f:
                    f.write(self.chat_template)

        config.save_pretrained(save_directory)

        import shutil
        vocab_dest = os.path.join(save_directory, os.path.basename(self.vocab_file))
        if os.path.abspath(self.vocab_file) != os.path.abspath(vocab_dest):
            shutil.copy(self.vocab_file, vocab_dest)


class AutoTokenizer:
    """AutoTokenizer - automatically selects and loads the appropriate tokenizer."""

    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> RWKVTokenizer:
        """
        Auto-load tokenizer from a pretrained model directory.

        Args:
            pretrained_model_name_or_path: Model directory path or model name.
            **kwargs: Extra arguments passed to the tokenizer.

        Returns:
            RWKVTokenizer instance.
        """
        return RWKVTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
