# 08a position ablation (teacher-forced PPL)

| condition | ppl |
|---|---|
| normal | 17.357 |
| reversed | 236.670 |
| shuffled | 489.776 |
| zero | 258.915 |

# 08c lexical unscrambling

| condition | similarity | kendall_tau | exact | scrambled_similarity |
|---|---|---|---|---|
| shuffle_0pct | 0.317 | 0.225 | 0.000 | 0.901 |
| shuffle_15pct | 0.502 | 0.443 | 0.000 | 0.867 |
| shuffle_30pct | 0.387 | 0.291 | 0.000 | 0.729 |
| shuffle_50pct | 0.142 | 0.172 | 0.000 | 0.597 |
