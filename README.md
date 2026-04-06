# o200ktok -- A Super Fast BPE Tokenizer

**A high-performance BPE tokenizer compatible with OpenAI's o200k_base vocabulary -- up to 29 times faster than tiktoken, the fastest tokenizer available today.**

> **29.2 times** faster than tiktoken (parallel mode)

---

## What Is a Tokenizer?

Before a large language model can read a single word you type, that text must be broken into **tokens** -- small chunks that map to numbers the model actually understands. Think of it as the translator sitting between human language and the neural network. The tokenizer splits your text into subwords, characters, or byte sequences using an algorithm called **Byte-Pair Encoding (BPE)**, then looks each piece up in a fixed vocabulary to produce a sequence of integer IDs.

This step is not optional. Every prompt, every document, every line of code -- it all passes through the tokenizer before the model sees it and again when the model's output is decoded back into text. A slow tokenizer becomes a bottleneck in data preprocessing, evaluation pipelines, and any workload that touches raw text at scale.

## Why Another Tokenizer?

OpenAI's `tiktoken` is the current gold standard for tokenizer performance. Written in Rust with Python bindings, it is the fastest tokenizer available today -- not just among BPE implementations, but across tokenizer libraries in general. As the SentencePiece benchmark later in this post shows, even a Rust-native SentencePiece implementation runs dramatically slower than tiktoken. When people benchmark tokenizers, tiktoken is the one to beat. Being faster than tiktoken means being faster than everything else.

The vocabulary in play here matters too. `o200k_base` is one of the largest and most comprehensive BPE vocabularies in production use -- 200,000 tokens, designed to cover a wide range of languages, code, and special characters. It powers GPT-4o and later OpenAI models. A larger vocabulary means more merge rules to evaluate during encoding, which makes fast tokenization harder, not easier. Achieving a 7 times speedup on a vocabulary this size is a different challenge than doing it on a smaller, simpler one.

`o200ktok` is a standalone CLI tokenizer built for heavy workloads -- data preprocessing, corpus analytics, batch evaluation. It implements the same BPE merge rules over the same `o200k_base` vocabulary, producing **bit-identical output** -- but it does so significantly faster than the tool that currently holds the performance crown. On a single thread it's up to 10.9 times faster; with the `--parallel` flag, which splits work across all available CPU cores, it reaches **29.2 times faster** than tiktoken on the same hardware.

## Benchmark

The test corpus is [WikiText-103 training set](https://huggingface.co/datasets/Salesforce/wikitext), a standard NLP benchmark dataset. Both tools were run on the same machine, tokenizing the full file and writing results to disk. Two modes were measured: *IDs-only* (output token IDs, one per line) and *Tokens* (output each ID with its decoded text value).

| | Value |
|---|---|
| **Tokens produced** | 119.2M |
| **Vocabulary size** | 200K |
| **Output match** | 100% |

### Single-Thread: IDs-Only Mode

| Tokenizer | Time | Comparison |
|---|---|---|
| **o200ktok** | **14.5s** | -- |
| tiktoken | 1m 50.5s | 7.6 times slower |

### Single-Thread: Tokens + Text Mode

| Tokenizer | Time | Comparison |
|---|---|---|
| **o200ktok** | **16.8s** | -- |
| tiktoken | 3m 3.7s | 10.9 times slower |

### Parallel Mode: IDs-Only (multi-CPU)

| Tokenizer | Time | Comparison |
|---|---|---|
| **o200ktok** | **4.5s** | -- |
| tiktoken | 1m 50.5s | 24.4 times slower |

### Parallel Mode: Tokens + Text (multi-CPU)

| Tokenizer | Time | Comparison |
|---|---|---|
| **o200ktok** | **6.3s** | -- |
| tiktoken | 3m 3.7s | 29.2 times slower |

Here are the raw timing results -- you can reproduce these yourself. No cherry-picking, no warm caches, just `time` on the command line:

```
tkn@m1:> time o200ktok --tokens -ids-only /data/dt/wikitext103_train.txt > tknres/o200k_ids-only.txt

real    0m14.507s
user    0m13.038s
sys     0m1.589s

tkn@m1:> time o200ktok --tokens /data/dt/wikitext103_train.txt > tknres/o200k_tokens.txt

real    0m16.754s
user    0m13.689s
sys     0m3.207s
```

```
tkn@m1:> time python3 tokenizer.py -v o200k_base --tokens --ids-only -f /data/dt/wikitext103_train.txt > tknres/tiktoken_ids-only.txt

real    1m50.539s
user    1m44.232s
sys     0m6.271s

tkn@m1:> time python3 tokenizer.py -v o200k_base --tokens -f /data/dt/wikitext103_train.txt > tknres/tiktoken_tokens.txt

real    3m3.694s
user    2m52.503s
sys     0m11.072s
```

And with the `--parallel` flag, `o200ktok` splits the work across all available CPU cores -- notice how `user` time exceeds `real` time, confirming true multi-core utilization:

```
tkn@m1:> time o200ktok --tokens --parallel -ids-only -f /data/dt/wikitext103_train.txt > tknres/o200k_tokens_paral_ids.txt

real    0m4.533s
user    0m27.685s
sys     0m2.395s

tkn@m1:> time o200ktok --tokens --parallel -f /data/dt/wikitext103_train.txt > tknres/o200k_tokens_paral.txt

real    0m6.288s
user    0m28.681s
sys     0m4.872s
```

> ✅ All modes -- single-thread, parallel, IDs-only, tokens -- produced exactly 119,160,779 tokens with identical output.

## Correctness First

Speed means nothing if the output is wrong. As the benchmark confirms, `o200ktok` produces **byte-for-byte identical results** to tiktoken on the full WikiText-103 training set -- same token count, same token IDs, same decoded text. This holds in both single-thread and parallel mode. This isn't approximate compatibility; it's exact.

Let's look at the output side-by-side. First the decoded tokens:

```
tkn@m1:> head tknres/o200k_tokens.txt
198     '\n'
314     ' ='
142393  ' Valk'
131854  'yria'
109152  ' Chronicles'
18857   ' III'
314     ' ='
25980   ' \n\n\n'
8675    ' Sen'
73      'j'

tkn@m1:> head tknres/tiktoken_tokens.txt
198     '\n'
314     ' ='
142393  ' Valk'
131854  'yria'
109152  ' Chronicles'
18857   ' III'
314     ' ='
25980   ' \n\n\n'
8675    ' Sen'
73      'j'
```

Then the raw IDs:

```
tkn@m1:> head tknres/o200k_ids-only.txt
198
314
142393
131854
109152
18857
314
25980
8675
73

tkn@m1:> head tknres/tiktoken_ids-only.txt
198
314
142393
131854
109152
18857
314
25980
8675
73
```

And the parallel mode? Same result -- splitting work across cores doesn't affect correctness:

```
tkn@m1:> head tknres/o200k_tokens_paral.txt
198     '\n'
314     ' ='
142393  ' Valk'
131854  'yria'
109152  ' Chronicles'
18857   ' III'
314     ' ='
25980   ' \n\n\n'
8675    ' Sen'
73      'j'

tkn@m1:> head tknres/o200k_tokens_paral_ids.txt
198
314
142393
131854
109152
18857
314
25980
8675
73
```

And the final proof -- `wc` confirms every line, word, and byte matches exactly:

```
tkn@m1:> wc tknres/o200k_ids-only.txt
 119160779  119160779  578442893 tknres/o200k_ids-only.txt

tkn@m1:> wc tknres/tiktoken_ids-only.txt
 119160779  119160779  578442893 tknres/tiktoken_ids-only.txt

tkn@m1:> wc tknres/o200k_tokens.txt
 119160779  340912896 1480084708 tknres/o200k_tokens.txt

tkn@m1:> wc tknres/tiktoken_tokens.txt
 119160779  340912896 1480084708 tknres/tiktoken_tokens.txt

tkn@m1:> wc tknres/o200k_tokens_paral.txt
 119160779  340912896 1480084708 tknres/o200k_tokens_paral.txt

tkn@m1:> wc tknres/o200k_tokens_paral_ids.txt
 119160779  119160779  578442893 tknres/o200k_tokens_paral_ids.txt
```

## Bonus: SentencePiece Benchmark

BPE isn't the only tokenization algorithm in production. Google's **SentencePiece** is one of the most widely adopted tokenizer frameworks in the LLM ecosystem. It powers models from virtually every major AI lab: **Google** (Gemini, Gemma, PaLM, T5), **Meta** (LLaMA 1 & 2), **xAI** (Grok-1), **Mistral**, and the multilingual **BLOOM** model -- among many others. If you work with LLMs, there's a good chance you're running SentencePiece tokenization somewhere in your stack.

To test whether the same performance principles apply, I built `sentence-piece-tok` -- a SentencePiece-compatible tokenizer using the Gemma 4 vocabulary (262,144 tokens, one of the largest SentencePiece vocabularies in production).

For comparison, I benchmarked against `sptok`, a Rust-based SentencePiece implementation. The result was surprising -- but also confirms tiktoken's position as the current performance champion: `sptok`, despite being written in Rust, took over 12 minutes to tokenize the WikiText-103 corpus. My `sentence-piece-tok` completed the same job in just 25 seconds with parallel mode -- a 28.5 times speedup.

| | Value |
|---|---|
| **Tokens produced** | 121.6M |
| **Vocabulary size** | 262K |
| **Output match** | 100% |

### IDs-Only: Single-Thread

| Tokenizer | Time | Comparison |
|---|---|---|
| **sentence-piece-tok** | **60.8s** | -- |
| sptok (Rust) | 12m 11.7s | 12 times slower |

### IDs-Only: Parallel (multi-CPU)

| Tokenizer | Time | Comparison |
|---|---|---|
| **sentence-piece-tok** | **25.7s** | -- |
| sptok (Rust) | 12m 11.7s | 28.5 times slower |

Here are the raw results. Note the extreme system time for `sptok` -- over 9 minutes of the 12-minute runtime is spent in kernel overhead:

```
tkn@m1:> cat /data/dt/wikitext103_train.txt | time ./sptok/target/release/sptok -f sptok/tokenizer.json encode > tknres/sp_rust_ids.txt
Loaded 262144 vocab + 514906 merges in 594ms
Encoded 541096899 chars -> 121611599 tokens in 86831.941ms
136.21user 594.59system 12:11.69elapsed 99%CPU
0inputs+1230776outputs (0major+881096minor)pagefaults 0swaps
```

```
tkn@m1:> cat /data/dt/wikitext103_train.txt | time ./sentence-piece-tok --ids-only > tknres/sp_loc_ids.txt
59.34user 2.06system 1:00.77elapsed 101%CPU
0inputs+1230464outputs (0major+470599minor)pagefaults 0swaps

tkn@m1:> cat /data/dt/wikitext103_train.txt | time ./sentence-piece-tok --ids-only -parallel > tknres/sp_loc_ids_paral.txt
134.52user 8.49system 0:25.70elapsed 556%CPU
0inputs+1230448outputs (0major+4524752minor)pagefaults 0swaps
```

And as always, the output is identical across all three runs:

```
tkn@m1:> wc tknres/sp_rust_ids.txt
 121611599  121611599  629986416 tknres/sp_rust_ids.txt

tkn@m1:> wc tknres/sp_loc_ids.txt
 121611599  121611599  629986416 tknres/sp_loc_ids.txt

tkn@m1:> wc tknres/sp_loc_ids_paral.txt
 121611599  121611599  629986416 tknres/sp_loc_ids_paral.txt
```

> ✅ All three runs produced exactly 121,611,599 tokens with identical output -- line count, word count, byte count all match.

## Usage

`o200ktok` is a single-binary CLI tool. No Python environment, no pip install, no dependency resolution -- just download and run.

```
tkn@m1:> o200ktok --help
Usage: o200ktok [flags] [file ...]

Flags:
  -batch        read file paths from stdin, one per line
  -f string     file to tokenize
  -ids-only     output token ids only, one per line
  -parallel     use multiple CPUs for large inputs
  -summary      print summary (token count, vocab info)
  -text string  text to tokenize
  -tokens       output id<TAB>value per line

Examples:
  echo "Hello world" | o200ktok
  o200ktok -f input.txt
  o200ktok --text "Hello world"
  o200ktok --text "Hello world" --ids-only
  cat big.txt | o200ktok --tokens | head
  o200ktok --text "Hello world" --summary

Batch mode (load once, process many files):
  o200ktok --tokens file1.txt file2.txt file3.txt
  find /data -name "*.txt" | o200ktok --batch --tokens
  o200ktok --batch --summary < filelist.txt
```
