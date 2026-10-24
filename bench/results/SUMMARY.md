# Raised benchmark

RMS-F1: DePlot's table metric (arXiv 2212.10505) in percent, a failed chart scores 0. Median err: median |read - true| / value-axis range over every true value that was read (percent); `within 2%` and `missing` count all true values, so a missing value is a miss. Numeric metrics need the generator's exact axis, so only synthetic splits have them. type / series / scale acc: share of charts whose chart type, series count and log-or-linear value axis the model read correctly (configs that read semantics with a model only).

## synth-bench

| config | n | failed | RMS-F1 | median err (% of range) | within 2% | missing | type acc | series acc | scale acc | $ / figure | s / figure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel_only | 300 | 0 | 89.8 | 0.24 | 83.1% | 4.6% | - | - | - | 0.0000 | 1.72 |

By chart type:

| config | chart type | n | failed | RMS-F1 | median err (% of range) | within 2% | missing | type acc | series acc | scale acc | $ / figure | s / figure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel_only | bar | 56 | 0 | 98.6 | 0.16 | 97.9% | 1.2% | - | - | - | 0.0000 | 2.18 |
| pixel_only | grouped_bar | 51 | 0 | 85.5 | 0.18 | 86.5% | 5.2% | - | - | - | 0.0000 | 2.34 |
| pixel_only | hbar | 44 | 0 | 89.9 | 0.10 | 90.1% | 9.5% | - | - | - | 0.0000 | 1.47 |
| pixel_only | line | 99 | 0 | 83.4 | 0.31 | 78.6% | 4.7% | - | - | - | 0.0000 | 1.45 |
| pixel_only | scatter | 50 | 0 | 96.5 | 0.14 | 95.4% | 3.4% | - | - | - | 0.0000 | 1.36 |

## synth-probe

| config | n | failed | RMS-F1 | median err (% of range) | within 2% | missing | type acc | series acc | scale acc | $ / figure | s / figure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel_only | 50 | 0 | 92.9 | 0.19 | 87.3% | 4.3% | - | - | - | 0.0000 | 1.33 |

By chart type:

| config | chart type | n | failed | RMS-F1 | median err (% of range) | within 2% | missing | type acc | series acc | scale acc | $ / figure | s / figure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pixel_only | bar | 12 | 0 | 98.3 | 0.19 | 98.5% | 1.5% | - | - | - | 0.0000 | 1.42 |
| pixel_only | grouped_bar | 5 | 0 | 94.0 | 0.14 | 92.9% | 7.1% | - | - | - | 0.0000 | 2.39 |
| pixel_only | hbar | 11 | 0 | 99.8 | 0.10 | 100.0% | 0.0% | - | - | - | 0.0000 | 1.23 |
| pixel_only | line | 13 | 0 | 86.0 | 0.21 | 85.3% | 1.8% | - | - | - | 0.0000 | 1.13 |
| pixel_only | scatter | 9 | 0 | 86.8 | 0.11 | 82.8% | 17.2% | - | - | - | 0.0000 | 1.00 |
