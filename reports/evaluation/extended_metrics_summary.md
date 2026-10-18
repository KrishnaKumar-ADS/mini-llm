# Extended Mini LLM Generation Metrics

Samples analyzed: 20

| Metric | 25k | 100k |
|---|---:|---:|
| word_count | 155.5000 | 166.9000 |
| mean_sentence_words | 9.4817 | 8.8320 |
| distinct_1 | 0.5367 | 0.5105 |
| distinct_2 | 0.8722 | 0.8702 |
| distinct_3 | 0.9538 | 0.9529 |
| repeated_sentence_count | 0.2000 | 0.2000 |
| Fraction without terminal punctuation (heuristic) | 0.5000 | 0.5000 |

## Interpretation notes

- Distinct-n measures lexical diversity of n-grams, not meaning.
- Repeated-sentence count detects exact normalized repeats.
- Missing terminal punctuation is only a warning; it does not prove that a generation was truncated.
- Compare these diagnostics alongside validation loss and perplexity, not as a single overall quality score.
- The models used different training batch sizes, so the comparison does not isolate dataset size alone.
