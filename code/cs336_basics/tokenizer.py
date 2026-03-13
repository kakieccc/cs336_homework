"""
引入regex的BPE实现
在进行BPE合并之前，先用正则表达式把文本切成几大类。BPE只能在每一类内部进行合并，不允许跨越Regex的边界
"""

import regex as re
import unicodedata

# 分块风格
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""工具类方法"""
def get_stats(ids, counts=None):
    """
    统计整数序列中所有相邻二元组（pair）的出现次数
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    把所有连续出现的pair替换成新的token
    pair：二元组
    idx：新token的id
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def render_token(t: bytes) -> str:
    # 把token的字节序列渲染成可读字符串。
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

def replace_control_characters(s: str) -> str:
    # 把控制字符转义，避免直接打印造成显示干扰
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

class RegexTokenizer():

    def __init__(self):
        self.merges = {} # (int, int) -> int
        # 初始化特殊token映射及其反向映射 e.g.终止符等等
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.pattern = GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.byte_encoder = {i: i for i in range(256)}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, input_path, vocab_size, special_tokens):
        # 计算需要合并的次数
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 根据分块风格对原始文本进行处理
        text_chunks = re.findall(self.compiled_pattern, input_path)

        # 把每个文本块转换成字节id列表的列表
        # 每个文本块里面都是一个List,所有文本块又构成一个List
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 迭代合并
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}  #初始化词表，把所有字节值填充到对应位置
        for i in range(num_merges):
            stats = {}
            # 遍历所有chunk，把每个chunk的相邻pair频次累加到同一个stats字典里
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            # 找频次最大的pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # 更新ids，把每个chunk内所有连续出现的这个pair替换成新id
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 记录合并规则
            merges[pair] = idx
            # 扩展词表
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # 保存
        self.merges = merges 
        self.vocab = vocab
        self.register_special_tokens(special_tokens)

    def register_special_tokens(self, special_tokens):
        if special_tokens is None:
            mapping = {}
        elif isinstance(special_tokens, list):
            start = max(self.vocab.keys(), default=255) + 1
            mapping = {tok: start + i for i, tok in enumerate(special_tokens)}
        elif isinstance(special_tokens, dict):
            mapping = dict(special_tokens)
        else:
            raise TypeError
        # 更新全局special_tokens
        self.special_tokens = mapping   
        # 更新反向映射
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        # 重建词表
        # 仅把special写入已有vocab，避免破坏外部注入的GPT-2vocab
        for tok, idx in self.special_tokens.items():
            self.vocab[idx] = tok.encode("utf-8")

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        # 把解码后的part_bytes拼接
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """
        统计当前相邻pair
        用min找最早被合并的pair（优先级越高）
        如该pair不在merges,结束
        否则执行替换，继续下一轮
        """
        ids = [self.byte_encoder[b] for b in text_bytes]
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """
        用regex对文本切块
        每一块都分别调用encode_chunk进行编码
        拼接所有块的结果
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="all"):
        special = None
        # 识别special_tokens
        if allowed_special == "all":
            special = self.special_tokens
        # 不识别special_tokens
        elif allowed_special == "none":
            special = {}
        # 识别到special_tokens报错
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        # 只识别集合内的special_tokens
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        # 没有special_tokens的话直接分块解码
        if not special:
            return self.encode_ordinary(text)
        # 否则用re.split将文本分块
        special_pattern = "(" + "|".join(re.escape(k) for k in sorted(special, key=len, reverse=True)) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
    
    def encode_iterable(self, iterable, allowed_special="all"):
        for text in iterable:
            yield from self.encode(text, allowed_special=allowed_special)

    def _build_vocab(self):
        # 构建词表
        # 放入0-255单字节token
        # 按merges把新token拼接为左右子token的字节串
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
        self.compiled_pattern = re.compile(self.pattern)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.byte_encoder = {i: i for i in range(256)}