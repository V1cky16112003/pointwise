# Pointwise token consumption

Tokenizer: `Qwen/Qwen3-1.7B-Base`. Reproduce: `python -m scripts.count_tokens --heldout <heldout.jsonl>`.

## One state, k questions (Phase 6 latency request: ~524-token state, three-option choice questions)

| k | Pointwise, one pass | Pointwise, state cached | Generate-and-parse prompts | x fewer (one pass) | x fewer (cached) |
|---|---|---|---|---|---|
| 1 | 542 | 18 | 544 | 1.0x | 30.2x |
| 2 | 561 | 37 | 1090 | 1.9x | 29.5x |
| 4 | 603 | 79 | 2182 | 3.6x | 27.6x |
| 8 | 680 | 156 | 4365 | 6.4x | 28.0x |

## Held-out sources (one question per record; Pointwise's tokens only, generate-and-parse was not counted here)

| Source | Options | State tokens | Question + option tokens | Total | Question + options share |
|---|---|---|---|---|---|
| banking77 | 77 | 13.6 | 435.0 | 448.6 | 97% |
| emotion | 6 | 20.6 | 30.0 | 50.6 | 59% |
| paws | 2 | 62.5 | 17.0 | 79.5 | 21% |
| qnli | 2 | 53.9 | 18.0 | 71.9 | 25% |
| tweet_sentiment | 3 | 24.6 | 19.0 | 43.6 | 44% |
