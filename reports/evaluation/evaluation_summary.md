# Mini LLM Evaluation Summary

## Generation statistics

| Metric | 25k | 100k |
|---|---:|---:|
| Sample count | 10.0000 | 10.0000 |
| Mean word count | 155.5000 | 166.9000 |
| Mean sentence count | 18.9000 | 19.4000 |
| Mean unique-word ratio | 0.5367 | 0.5105 |
| Mean repeated-trigram ratio | 0.0462 | 0.0471 |
| Mean repeated sentences | 0.2000 | 0.2000 |

## Validation results

| Metric | 25k | 100k |
|---|---:|---:|
| Best validation loss | 0.7277123663490389 | 0.6862436973681224 |
| Perplexity at best-loss step | 2.0703390090356906 | 1.986240582216629 |
| Latest validation loss | 0.7277123663490389 | 0.6862436973681224 |
| Latest validation perplexity | 2.0703390090356906 | 1.986240582216629 |

## Limitations

- Automatic text statistics do not establish semantic coherence.
- Manually review the paired generations for fluency, logical consistency, repetition and relevance to the prompt.
- Training batch sizes differed (288 versus 1280), so this is not a controlled dataset-size-only comparison.
- Validation results are comparable only if the validation data, tokenizer and evaluation procedure match.
- Generation runtime includes process startup and checkpoint loading; it is not a clean throughput benchmark.
