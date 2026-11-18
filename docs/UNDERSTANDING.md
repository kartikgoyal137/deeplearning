# Understanding this project from scratch

This document explains, assuming no prior knowledge, what Mamba is, what the two variations change, and what the
evaluation code does. After reading it you should be able to follow the code without reading it first, and to explain the
project to someone else. Numbers in examples are real values from the code unless marked "toy".

**Contents**
1. [The problem: remembering the past](#1-the-problem-remembering-the-past)
2. [The key idea: a state that is updated token by token](#2-the-key-idea-a-state-that-is-updated-token-by-token)
3. [What "selective" means](#3-what-selective-means)
4. [One Mamba layer, step by step](#4-one-mamba-layer-step-by-step)
5. [The whole model](#5-the-whole-model)
6. [How training differs from generating (the scan)](#6-how-training-differs-from-generating-the-scan)
7. [Why copying is hard for Mamba](#7-why-copying-is-hard-for-mamba)
8. [The two tasks we test on](#8-the-two-tasks-we-test-on)
9. [Variation 1: state size changes with depth](#9-variation-1-state-size-changes-with-depth)
10. [Variation 2: state size grows along the sequence](#10-variation-2-state-size-grows-along-the-sequence)
11. [How the code is organised](#11-how-the-code-is-organised)
12. [Glossary](#12-glossary)

---

## 1. The problem: remembering the past

A language model reads a sequence of **tokens** (words, characters, numbers) and predicts the next one. To predict well it must
remember what came earlier. There are two classic ways to remember:

```
 TRANSFORMER (attention)                    RNN-style models (including Mamba)
 -----------------------                    ----------------------------------
 keeps EVERY past token around              keeps ONE fixed-size summary ("state")
 and looks back at all of them              and updates it with each new token

 token 1 token 2 ... token t                token 1 --> [ state ] --> token 2 --> [ state ] ...
    ^       ^          ^                                    ^
    +-------+----------+---- look back                      +-- the only memory there is
 
 memory grows with length                   memory is constant
 cost per token grows with length           cost per token is constant
 (cost of whole text ~ L x L)               (cost of whole text ~ L)
```

| | Transformer | Mamba |
|---|---|---|
| Memory of the past | grows with every token | **fixed size** |
| Cost of reading a text of length L | about L x L | about L |
| Strength | can look up anything exactly | fast and cheap |
| Weakness | expensive on long texts | the fixed memory can overflow |

This project is about that weakness and one idea for reducing it: **what if Mamba's memory could grow when it needs to?**

---

## 2. The key idea: a state that is updated token by token

Mamba belongs to a family called **state space models (SSMs)**. The "state" is a list of numbers `h` that acts as the memory.
For every new token `x_t` three things happen:

```
        old memory                  new token
          h(t-1)                      x(t)
             |                          |
             v                          v
     (1) FORGET a little      (2) WRITE the new token in
         h * decay                  + write * x
             \________________________/
                          |
                          v
                      new memory h(t)
                          |
                          v
                (3) READ an output:  y(t) = read * h(t)
```

In symbols (one number of memory, one channel):

```
h(t) = decay * h(t-1) + write * x(t)          # update the memory
y(t) = read * h(t)                             # produce an output
```

**A tiny worked example (toy numbers).** Let `decay = 0.5`, `write = 1`, `read = 1`, and the input be `4, 0, 0, 2`:

```
t=1   h = 0.5*0   + 1*4 = 4        the 4 is written
t=2   h = 0.5*4   + 1*0 = 2        it fades by half every step
t=3   h = 0.5*2   + 1*0 = 1
t=4   h = 0.5*1   + 1*2 = 2.5      a new value arrives on top of the faded old one
```

So memory is a **running, fading summary** of the past. A decay close to 1 means long memory; close to 0 means forgetting fast.

### The state is bigger than one number
Real Mamba has many **channels** (`d_inner`, e.g. 768) and gives **each channel its own `N` numbers** of memory (e.g. `N = 16`).
The `N` numbers of one channel forget at *different speeds* so that some remember short stretches and some long ones:

```
one channel's state = N numbers, each with its own forgetting speed

 dim 1   dim 2   dim 3   ...   dim N
 slow    ...                   fast        (forgetting speed, set by a fixed number A = -1, -2, ..., -N)
 forgets                       forgets
 little                        quickly

 example with step size 0.1: dim 1 keeps 90% per token, dim 16 keeps only 20% per token
```

```
total memory of one layer = d_inner channels  x  N numbers   =  768 x 16 = 12,288 numbers
```

**`N` is the "state size". Everything in this project is about `N`.**

---

## 3. What "selective" means

In older state space models `decay`, `write` and `read` were fixed constants. Mamba's innovation is that they are
**computed from the current token**, so the model chooses what to remember:

```
                          token x(t)
                              |
          +-------------------+--------------------+
          v                   v                    v
   Delta(t) = how       B(t) = what to        C(t) = what to
   much to forget/      write into the        read out of
   overwrite            memory                the memory
          |                   |                    |
          +--------> update the memory h <---------+----> output y(t)
```

The names used in the code:

| Name | Meaning | Where it comes from |
|---|---|---|
| `A` | fixed forgetting speeds (one per memory number), always negative | learned once, not per token |
| `Delta` (`dt`) | step size, "how strongly to apply the update at this token" | computed from the token |
| `B` | what to write | computed from the token |
| `C` | what to read | computed from the token |
| `D` | a direct "skip" from input to output | learned once per channel |

The exact update used in the code (per channel, for each of the `N` memory numbers):

```
decay(t)  = exp( Delta(t) * A )                       # between 0 and 1 because A < 0 and Delta > 0
h(t)      = decay(t) * h(t-1)  +  Delta(t) * B(t) * x(t)
y(t)      = C(t) . h(t)  +  D * x(t)                  # "." = sum over the N memory numbers
```

Because these depend on the token, the model can, for example, write strongly for important tokens and ignore filler.

---

## 4. One Mamba layer, step by step

A "layer" (called a **Mamba block** in the code) wraps the memory update with a few helpers. Real sizes from the
Shakespeare notebooks: model width `D = 384`, inner width `d_inner = 2*D = 768`, state size `N = 16`.

```
input to the layer: one vector of D=384 numbers per token
        |
        v
  [ in_proj ]        a linear map  384 -> 1536, then split in two halves:
        |                 x  (768 numbers)   the "content"
        |                 z  (768 numbers)   the "gate" (decides what to let through)
        v
  [ causal conv ]    each channel is mixed with its previous 3 tokens (window of 4).
        |            "causal" = never looks at future tokens. Gives cheap local context.
        v
  [ SiLU ]           a smooth on/off-like non-linearity
        |
        v
  [ SELECTIVE SSM ]  the memory update of section 3, one memory (768 x N) per layer.
        |            computes Delta, B, C from x with a small linear layer ("x_proj"),
        |            runs the update over the tokens, produces y.
        v
  [ x SiLU(z) ]      multiply by the gate: z decides how much of y passes
        |
        v
  [ out_proj ]       a linear map  768 -> 384 back to the model width
        |
        v
output: one vector of 384 numbers per token
```

Why each part exists: the conv gives local context, the SSM gives long-range memory, the gate lets the layer ignore its own
output when it is not useful, and the two linear maps move between the model width and the (wider) inner width.

**Memory carried by this layer while reading a text:** the `768 x N` state plus the last 4 inputs of the conv
(`768 x 4`) = 15,360 numbers for `N = 16`. It does not grow with the text length.

---

## 5. The whole model

Layers are stacked, and each has its **own** memory and its own parameters:

```
token ids (e.g. 'h','e','l','l','o')
        |
        v
  [ embedding ]                  turns each token id into a vector of D numbers
        |
        v
  +--------------------------------------------------------------+
  | layer 1:   x  <-  x + MambaBlock( RMSNorm(x) )   <- own memory |
  | layer 2:   x  <-  x + MambaBlock( RMSNorm(x) )   <- own memory |
  | layer 3:   ...                                                 |
  | layer 4:   ...                                                 |
  | layer 5:   ...                                                 |
  | layer 6:   x  <-  x + MambaBlock( RMSNorm(x) )   <- own memory |
  +--------------------------------------------------------------+
        |
        v
  [ RMSNorm ] -> [ output layer ]   vector of D numbers -> a score for every possible next token
        |                           (shares its weights with the embedding: "tied")
        v
  predicted probabilities for the next token
```

- **Residual connection** (`x + ...`): each layer adds its result to its input instead of replacing it, which keeps training stable.
- **RMSNorm**: rescales a vector to a standard size before each layer (also for stability).
- The layers are in **depth**; the memory inside each layer is over **time** (tokens). Two different directions:

```
            time (tokens) ------------------------------------->
 depth       t=1      t=2      t=3      ...
   |       layer 1 -> layer 1 -> layer 1 ...   <- layer 1's memory flows along time
   |          |          |          |
   v       layer 2 -> layer 2 -> layer 2 ...   <- layer 2's memory flows along time
              ...
```

- Total memory while reading: `6 layers x (768 x 16 + 768 x 4) = 92,160` numbers for the notebook model, independent of the text length.
- Parameter count of the notebook model: 5.81 million.

---

## 6. How training differs from generating (the scan)

**Generating text** is naturally one token at a time: feed a token through the 6 layers, update each layer's memory, predict
the next token, repeat. That is the cheap constant-memory mode.

**Training** has the whole text available at once, so we want to process all tokens in parallel on the GPU. But the memory
update looks sequential (`h(t)` needs `h(t-1)`). Mamba's trick is that this kind of update can be computed in parallel
with a **scan** (a running calculation using cumulative sums). Our code does it in **chunks of 8 tokens**:

```
tokens:   [ 1 2 3 4 5 6 7 8 ] [ 9 ... 16 ] [ 17 ... 24 ] ...
             chunk 1             chunk 2       chunk 3

 inside a chunk: all 8 tokens are processed at once (cumulative sums, a few doubling steps)
 between chunks: the memory h at the end of one chunk is carried into the next
```

This matters for cost: the scan creates big temporary tensors of shape `(batch, length, d_inner, N)`, so **GPU memory and
time grow with `N`** (we measured that in this implementation the state-dependent part is 80 to 100% of the step time). The official
Mamba hides this with a custom GPU kernel; ours is plain PyTorch, which is slower and heavier but easy to read and change.

The evaluation harnesses use the parallel mode only ("teacher forcing", see section 8).

---

## 7. Why copying is hard for Mamba

Task: given a random string, repeat it exactly.

```
  input:   $ h b u z g b b p m |      <- read this
  output:                        h b u z g b b p m .   <- reproduce it
```

To copy `n` symbols the model must keep all of them in its memory until it is time to write them out.
A Transformer can simply look back at the original tokens. Mamba has only its fixed-size state, so past some length the
memory is full:

```
 symbols that must be remembered
   ^
   |                                    Transformer: always able (memory grows with the text)
   |                              ...
   |                    ...
   |       ______________________________ Mamba: capacity limit (set by N, the number of channels, depth)
   |      /   
   |_____/____ fine below the limit,     fails above it
   +-------------------------------------> length of the string
```

The paper this project follows (`cot.pdf`) proves this limit and shows experiments where Mamba struggles as strings get longer
or as the model gets smaller. **The project's idea**: instead of one fixed memory size for the whole text, make the memory
bigger as the text gets longer, so the cost stays low for short stretches but the capacity is there when needed.

---

## 8. The two tasks we test on

Both come from published code that we adapted (see `copy_eval/README.md`, `cot_eval/README.md`). In both, the model is trained to
predict the next token and is then scored on how exactly it reproduces the right answer.

### 8.1 Copy task (`copy_eval/`)
Special symbols: `$` start, `|` separator, `.` end, `*` padding. Letters are the string itself.

```
 one example:   $ h b u z g | h b u z g .
                  \_______/   \_______/
                  the string   the answer (must equal the string)
```

- **Training:** many random strings of length 5 to 40. The loss is computed **only on the answer part and the end symbol**
  (the first half is random, so predicting it teaches nothing).
- **One example per context** (`--no_pack`): the example is followed by padding. Upstream code packs several examples into one
  context; we avoid that because var2's state size depends on the position in the context.
- **Scoring (teacher forcing):** the model is fed the correct example and at each position we check its prediction for the
  next token. `str_acc` = fraction of examples where the *whole* answer is right; `char_acc` = fraction of correct characters.
  This is done for each string length separately, so the result is a curve "accuracy versus string length".

### 8.2 Chain-of-thought task: longest increasing subsequence (`cot_eval/`)
The input is a list of numbers; the answer is the length of the longest strictly increasing subsequence.

```
 numbers:                3  1  4  1  5            LIS ending at each position:  1  1  2  1  3     answer: 3

 DIRECT version:   3 1 4 1 5 <sep> 3                              (jump straight to the answer)
 CHAIN-OF-THOUGHT: 3 1 4 1 5 <sep> 1 1 2 1 3 <sep> 3              (write the working out first)
```

The chain-of-thought version gives the model a **scratchpad** (the intermediate values) but makes the sequence about twice as
long, which is exactly where a fixed-size memory may struggle. A sequence counts as correct only if **every** token after the
first `<sep>` (all steps and the final answer) is predicted correctly.

---

## 9. Variation 1: state size changes with depth

**Standard Mamba** uses the same `N` in every layer. **var1** keeps the *total* fixed and spreads it unevenly: small `N` in
early layers, large `N` in late layers.

```
 6 layers, mean N = 16, total = 96

            standard (N = 16 everywhere)        var1 "linear"
 layer 1    ################ 16                 ##### 5
 layer 2    ################ 16                 ######### 9
 layer 3    ################ 16                 ############# 13
 layer 4    ################ 16                 ################### 19
 layer 5    ################ 16                 ####################### 23
 layer 6    ################ 16                 ########################### 27
 total      96                                   96   (same parameters, same total memory)
```

Hypothesis: early layers mostly do local work (the convolution already covers it) and do not need much memory, while late layers
must carry long-range information. In this variation the memory size of each layer is **fixed over time**; only the
layers differ. It is kept exactly as originally written in `var1/` (a separate module, selected with `--model mamba_var1`).

---

## 10. Variation 2: state size grows along the sequence

**var2** (`mamba_growing.py`) changes `N` over *time* instead of depth. The state starts small and gets bigger at chosen token
positions, up to a cap. The schedule is written as `start_token:N` pairs:

```
 schedule  "0:8,16:16,32:32,64:64"

 N (state size)
 64 |                                              ┌─────────────────  capped here: memory is bounded
 32 |                              ┌───────────────┘
 16 |               ┌──────────────┘
  8 |───────────────┘
    +---------------+--------------+---------------+-----------------> token position t
    0               16             32              64
```

The same schedule is used in every layer, and each layer has its own growing memory.

### What happens when the size grows
The new memory numbers are **added with the value zero**, and the old ones are **kept as they are**. Nothing is erased
and nothing needs to be converted:

```
 before the switch at token 16:   h = [ a1 a2 a3 a4 a5 a6 a7 a8 ]                              (8 numbers)
 after the switch:                h = [ a1 a2 a3 a4 a5 a6 a7 a8 | 0  0  0  0  0  0  0  0 ]      (16 numbers)
                                      \____ kept exactly ____/    \__ new, empty __/
```

### How the code does it
The sequence is cut into **segments** where `N` is constant, and the segments are processed one after another, passing the
memory along. For a text of 70 tokens with the schedule above:

```
 tokens   0 ............. 15 | 16 ......... 31 | 32 ............ 63 | 64 .. 69
 N                8          |       16         |        32          |   64
 segment          1          |        2         |         3          |    4

 for each segment:
   1. take only the parameters for this N     (x_proj rows for Delta, B[0:N], C[0:N]; A[:, 0:N])
   2. grow the memory: zero-pad h from the old N to this N     (first segment: h = zeros)
   3. run the normal chunked scan over the segment's tokens, starting from h; keep the final h
 finally: put the outputs of all segments back together
```

Two details that make this efficient and correct:
- **Parameters are allocated for the largest `N`** (the cap), because a late stage may use all of them. An early segment just uses
  the first `N` of each parameter row. This is why var2 has more parameters than a standard `N = 16` model
  (about 11 to 16% more) and the *same* number as a standard model with `N = cap`.
- **Cost follows the schedule**: in a segment with a small `N` the scan and the projection only handle `N` numbers per channel,
  so short stretches are cheap. The saving is largest on short texts and shrinks as the text gets long and stays at the cap.

### What stays identical to standard Mamba
Embedding, norms, convolution, gating, in/out projections, the output layer, and the update formula of section 3. Only the
size of the memory `h` over time changes. In the code: `GrowingMambaLM` builds a normal model and replaces each layer's SSM by a
`GrowingSSM` (the file `mamba_growing.py`, about 100 lines, one function copied from the standard scan with a small change so it
can start from a given memory and return the final memory).

### What is tested (`test_growing.py`)
- A schedule with a single stage gives exactly the same output as standard Mamba.
- A multi-stage schedule gives the same output as a reference that always uses the full memory size but sets the
  write/read of the not-yet-active positions to zero.
- The model is causal (changing a token never changes earlier outputs), also across a switch.
- All parameters receive gradients, including the ones that only become active late.

### Limits
No token-by-token generation yet (training and scoring only); `--compile` (the fast GPU compile option) is not supported
for var2; the cost savings have been designed but not yet benchmarked, and there are no accuracy results yet.

---

## 11. How the code is organised

```
 deeplearning/
 ├─ mamba_model.py            STANDARD Mamba. Classes, bottom to top:
 │                              RMSNorm -> SelectiveSSM (the memory, section 3) -> MambaBlock (section 4)
 │                              -> ResidualBlock -> MambaLM (the whole model, section 5)
 │                            plus the scan functions of section 6
 ├─ 01_mamba_train.ipynb      trains standard Mamba on Shakespeare text; 02_... generates text with it
 ├─ var1/                     VARIATION 1, kept exactly as originally written (model + its two notebooks)
 ├─ mamba_growing.py          VARIATION 2 (section 10)
 ├─ test_growing.py           tests for variation 2
 ├─ copy_eval/                COPY task: train + score any of the three models      (section 8.1)
 │    ├─ data_utils.py          makes the examples and the loss mask   (upstream code)
 │    ├─ train_utils.py         the training loop                       (upstream code, small edits)
 │    ├─ test_utils.py          the scoring: str_acc / char_acc         (upstream code, small edits)
 │    ├─ model_utils.py         builds the models: standard, var1, var2 (+ same initialisation as official Mamba)
 │    ├─ main.py                the command-line entry point
 │    └─ sanity_checks.py       tests of the data, the metrics and the model
 ├─ cot_eval/                 CHAIN-OF-THOUGHT task (LIS): train + score any of the three models   (section 8.2)
 │    ├─ LIS/data.py            generates the number lists and their working out   (upstream code)
 │    ├─ LIS/dataloader.py      turns them into tokens and the loss mask           (upstream code)
 │    ├─ LIS/eval.py            the scoring                                         (upstream code)
 │    ├─ train.py, test.py      training and testing                                (upstream, small edits)
 │    └─ model_utils.py         builds the models (uses copy_eval's builder)
 └─ docs/UNDERSTANDING.md     this file
```

**Choosing the model** in either harness: `--model mamba` (standard), `--model mamba_var1` (with `--state_schedule constant|linear`),
`--model mamba_var2` (with `--schedule "0:8,16:16,32:32,64:64"`). Everything else (data, loss, optimiser, scoring) is
identical for the three, so differences in the results come from the memory design only.

**The comparison we plan** (the experiment that decides whether var2 is worth it): for each string length, compare var2 with
standard Mamba of constant `N` = 8, 16, 32, 64. The question is at which lengths var2 reaches the accuracy of the large
constant models (same parameters as var2 at `N = 64`) while using less memory and compute on average, or beats a constant
model of the same average cost. No results exist yet.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| token | one unit of input (a character, number or word piece) |
| sequence length `L` | number of tokens in an example |
| model width `D` (`d_model`) | size of the vector that represents a token between layers (e.g. 384) |
| inner width `d_inner` | `2 x D`; the width inside a Mamba block; also the number of memory "channels" |
| state `h` | the memory of one layer: `d_inner x N` numbers |
| state size `N` | how many memory numbers each channel has. **The quantity this project varies.** |
| `A, Delta, B, C, D` | the SSM's forgetting speeds, step size, write, read and skip (section 3) |
| selective | `Delta, B, C` are computed from each token |
| layer / block | one `RMSNorm + Mamba block + residual`; the model stacks several |
| scan | the parallel way to run the memory update over many tokens at once |
| chunk | the 8 tokens the scan handles at once before carrying the state onward |
| teacher forcing | scoring by feeding the *correct* text and checking each next-token prediction |
| `str_acc` / `char_acc` | whole-answer-correct rate / per-character-correct rate |
| chain of thought (CoT) | writing intermediate steps before the answer |
| schedule | the `start_token:N` list that tells var2 how the state size grows |
| cap (`N_max`) | the largest `N` in the schedule; bounds the memory |
| baseline | standard Mamba with a constant `N`, the thing the variations are compared to |
