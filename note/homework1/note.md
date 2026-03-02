# homework1_Building a Transformer LM

[toc]

## Notes

### Tokenization(分词)

#### 相关概念

1. tokenization：将原始的文本信息（一般为Unicode字符串）映射到离散的 Tokens 列表的步骤。
2. token：LLM处理文本的基本单位，一个 Token 可以是一个完整的单词、一个词的一部分（如前缀或后缀）、一个标点符号，甚至是一个单个字符。
3. encode（编码）：将token转化为特定整数的过程，这个整数被称为code point，每一个code point都对应了一个独特的token。
4. decode（解码）：将整数数组重新转化成文本字符串的过程。
5. compression ratio（压缩率）：字符串的字节数 / token数，每个token能表示的字节数。

#### character-based tokenization

| 特征     | 描述                                                         |
| -------- | :----------------------------------------------------------- |
| 基本单位 | 单个 Unicode 字符（例如：'H', 'e', 'l', 'l', 'o', '🌍', '!'）。 |
| 编码方式 | 每个字符转换为其 Unicode 码点（整数）。例如 $ \text{ord}(a) = 97 $，$ \text{ord}(🌍) = 127757 $。 |
| 词汇表   | 非常大，因为大约有 150K 个 Unicode 字符。                    |
| 序列长度 | 很长，因为一个字符对应一个 Token，导致压缩率低。             |
| 存在问题 | 1. 词汇表太大。2. 许多字符（如 '🌍'）很罕见，导致对词汇表的使用效率低下。 |

#### Byte-based tokenization

| 特征     | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| 基本单位 | 单个字节（一个 0 到 255 之间的整数）。                       |
| 编码方式 | Unicode 字符串首先被编码为字节序列，最常用的是 UTF-8。例如，`"a"` 是一个字节，而 `"🌍"` 是多个字节（`b'\xf0\x9f\x8c\x8d'`）。 |
| 词汇表   | 小且固定，只有 256 个。                                      |
| 序列长度 | 很长，因为压缩率为 1（即一个 Token 只有一个字节）。序列太长会导致 Transformer 的注意力机制（通常是二次复杂度）效率低下。 |
| 存在问题 | 压缩率非常差，序列太长。                                     |

#### word-based tokenization

| 特征     | 描述                                                         |
| -------- | ------------------------------------------------------------ |
| 基本单位 | 完整的单词或通过正则表达式分割的语段。                       |
| 编码方式 | 将每个语段映射到一个整数 ID。                                |
| 词汇表   | 巨大，每一个单词/语段都需要被设置成一个单独的词汇。          |
| 存在问题 | 1. 词汇量巨大。2. 许多单词很罕见。3. 新词汇（训练时未见过的词）会得到一个特殊的 UNK (Unknown) Token，这会影响计算。 |

#### BPE（Byte-Pair Encoding ）

1994年被提出，在 2016 年被 Sennrich 等人引入自然语言处理（NLP）领域，用于神经机器翻译，并被 GPT-2 等现代 LLM 广泛采用。之前传统NLP所采用的分词方法基本都是word-based tokenization。

核心思想：通过在原始文本上训练 Tokenizer 来自动确定词汇表。它是一种有效的启发式方法，利用语料库统计数据来确定 Token。常用的字符序列用一个 Token 表示，罕见的序列用多个 Token 表示。

训练过程：从字符串的字节列表开始，重复合并出现频率最高的相邻 Token 对，每合并一次，就会产生一个新的 Token 索引，词汇表也随之扩大。

### Tensor(张量)

tensor 就是数据的多维数组，无论输入的是文字、图片还是声音，最终都会被转化成 tensor 供模型计算。

 tensor 与普通数组（如 NumPy）的区别：

1. GPU 加速： Tensor 可以运行在 GPU 上，计算速度比 CPU 快几十倍甚至上百倍。
2. 自动求导：Tensor 会记录自己的计算历史，使得模型在学习时，能自动计算梯度（误差）并更新参数。

初始化tensor的几种方式：

```python
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 直接初始化值
    x = torch.zeros(4, 8)  # 初始化4*8,全为0的张量
    x = torch.ones(4, 8)  # 初始化4*8,全为1的张量
    x = torch.randn(4, 8)  # 初始化4*8,取值在(0-1)的随机数的张量
    x = torch.empty(4, 8)  # 初始化4*8的张量，但不赋初值
```

在pytorch中，tensor实际上是指向已分配内存的一些指针，所以对tensor进行创建视图操作并不会创建一个新的tensor。

例如改变视图:

```python
x = torch.tensor([1,2,3],[4,5,6])
aay = x.view(3,2) #改变视图，把2*3的x矩阵看成3*2的y矩阵
```

x和y指向的是同一片内存区域，而不是复制x并在内存中开一个新的空间。所以对x的元素进行操作时会同时影响到y的对应元素。

除此之外还有以下操作，它们都是创建视图，不会创建副本：

- 切片：`b = a[0:5]`。取数组的一部分时，PyTorch 只会改变偏移量（Offset）和步长（Stride）。
- 形状重塑：使用 `.view()` 或 `.reshape()`（在满足连续性时）。这里的连续性指的是tensor中元数据的逻辑排列顺序和内存当中的物理排序顺序一致。
- 转置：`b = a.t()`。仅仅是交换了维度的步长信息。
- 增加/删除维度：`b = a.unsqueeze(0)`。数据本身没变，只变了元数据。

对tensor进行创建副本操作会开辟新的内存空间，两个tensor互不影响。

- 强制克隆：`b = a.clone()`。
- 转换为 NumPy 再转回：`b = torch.tensor(a.numpy())`。跨框架转换通常都会触发复制（但不总是）
- 不连续 Tensor 的重塑：一个 Tensor 经过了转置（不连续），再调用 `.reshape()`，框架会被迫创建副本以保证内存连续性。

- 索引访问：`b = a[a > 0]`：因为选出的元素在内存中可能不连续，必须重新排列。
- 跨设备移动：`.to()`, `.cuda()`。把tensor从cpu搬到gpu。

### einops

einops 是一个专门为深度学习设计的 Tensor 操作库，核心目标是把复杂的 Tensor 维度变换（如 `reshape`, `transpose`, `stack`, `view` 等）变得极其直观且易读。简单来讲就是把tensor操作里面的维度操作用语义变得更直观，而不是一个光秃秃的数字让你去猜这个维度代表的含义。

einops 主要由三个函数统治：

1. `rearrange` 

它可以代替 `view`, `reshape`, `transpose`, `permute`, `stack`, `split` 等所有改变维度的操作。

- 转置： `rearrange(x, 'b h w c -> b c h w')`（把通道移动到前面）
- 展平： `rearrange(x, 'b c h w -> b (c h w)')`
- 切块 (Patching)： 将一张大图切成小方块， `rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)`

2.  `reduce` 

它可以代替 `sum`, `mean`, `max` 等操作，并允许指定在哪个维度进行归约。

- 全局平均池化： `reduce(x, 'b c h w -> b c', 'mean')`
- 把彩色图转灰度图： `reduce(x, 'b c h w -> b h w', 'mean')`（对通道维度求平均）

3. `repeat` 

它可以代替 `expand` 或 `tile`。

- 增加维度并复制： `repeat(x, 'c h w -> b c h w', b=32)`（把一张图复制 32 份组成一个 Batch）

参数名一般遵循以下规则：

| 字母   | 全称 (Full Name)    | 含义                                                      |
| ------ | ------------------- | --------------------------------------------------------- |
| b      | Batch               | 批次大小。一次处理多少个样本。                            |
| c      | Channel             | 通道数。如 RGB 图像的 `3`，或者特征向量的维度。           |
| h, w   | Height, Width       | 图像或特征图的高度和宽度。                                |
| n      | Number / Sequence   | 序列长度。在 NLP 中常用 `n` 或 `s`（Sequence）。          |
| d      | Dimension / Feature | 每个词或像素对应的向量维度（Embedding Dimension）。       |
| t      | Time                | 时间步。常用于视频（帧数）或时间序列数据。                |
| p1, p2 | Patch               | 图像切块（Patch）的大小。                                 |
| g      | Group               | 组。在 Group Convolution 或 Multi-head Attention 中常用。 |

## Homeworks

### assignment

1. 在给定训练集上训练自制的BPE分词器
2. 运行BPE分词器，将给定数据转换
3. 实现一个基于Transformer的LM,并在给定数据集上训练
4. 使用训练好的 Transformer LM生成样本并评估困惑度
5. 在 OpenWebText 数据集上训练模型，并将所达到的困惑度结果提交

### implement

#### BPE Tokenizer

 
