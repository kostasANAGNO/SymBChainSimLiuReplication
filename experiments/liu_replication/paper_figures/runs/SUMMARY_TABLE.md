## Sweeps (endpoints; ours = DES, paper = read off figures)

### fig7 (throughput, x: 2→20 ω)

| scheme | paper | DES default links | DES hi links |
|---|---|---|---|
| proposed | 159.0k → 227.0k | 1.7k → 38.0k | 10.4k → 80.0k |
| no_cas | 155.0k → 221.0k | 0.7k → 24.1k | 7.2k → 80.0k |
| fixed_block_size | 147.0k → 184.0k | 2.7k → 39.0k | 13.3k → 40.0k |
| fixed_block_interval | 143.0k → 182.0k | 1.6k → 39.0k | 14.0k → 40.0k |
| static | 142.0k → 174.0k | 1.9k → 38.0k | 17.0k → 80.0k |

### fig8 (throughput, x: 100→550 χ)

| scheme | paper | DES default links | DES hi links |
|---|---|---|---|
| proposed | 212.0k → 131.0k | 11.3k → 4.8k | 96.0k → 29.1k |
| no_cas | 206.0k → 128.0k | 9.2k → 1.5k | 56.0k → 0.0k |
| fixed_block_size | 175.0k → 116.0k | 19.5k → 4.6k | 80.0k → 14.5k |
| fixed_block_interval | 172.0k → 114.0k | 28.5k → 3.9k | 80.0k → 14.5k |
| static | 158.0k → 107.0k | 26.4k → 5.2k | 108.0k → 29.1k |

### fig9 (ttf, x: 10→30 GHz)

| scheme | paper | DES default links | DES hi links |
|---|---|---|---|
| proposed | 5.80 → 3.30 | 13.93 → 13.18 | 2.83 → 2.83 |
| no_cas | 6.20 → 3.70 | 18.48 → 6.37 | 5.47 → 5.59 |
| fixed_block_size | 7.60 → 5.70 | 7.93 → 6.92 | 1.66 → 2.15 |
| fixed_block_interval | 7.80 → 6.30 | 4.55 → 5.39 | 4.38 → 3.33 |
| static | 8.60 → 7.20 | 15.25 → 14.36 | 2.78 → 2.83 |

### fig10 (throughput, x: 1→10 Ṡ)

| scheme | paper | DES default links | DES hi links |
|---|---|---|---|
| proposed | 6.1k → 9.3k | 10.0k → 11.6k | 10.0k → 80.0k |
| no_cas | 6.0k → 8.9k | 3.0k → 5.4k | 10.0k → 33.3k |
| fixed_block_size | 6.0k → 6.0k | 9.8k → 12.7k | 10.0k → 40.0k |
| fixed_block_interval | 5.0k → 7.2k | 4.9k → 12.3k | 5.0k → 50.0k |
| static | 4.8k → 7.0k | 9.5k → 15.3k | 10.0k → 40.0k |

## Fig. 5 plateau (common last-1000-episode window), TPS, relative to proposed

| scheme | paper | rel. | DES default | rel. | DES hi | rel. |
|---|---|---|---|---|---|---|
| proposed | 210k | 1.00 | 14.2k | 1.00 | 78.8k | 1.00 |
| no_cas | 165k | 0.79 | 5.7k | 0.40 | 34.5k | 0.44 |
| fixed_block_size | 125k | 0.60 | 12.7k | 0.90 | 37.7k | 0.48 |
| fixed_block_interval | 95k | 0.45 | 14.2k | 1.00 | 39.5k | 0.50 |

## Fig. 9 alpha/beta sensitivity (hi links): TTF in s at 10 -> 30 GHz, change, and paper

| scheme | paper | hi baseline (alpha, beta per block) | V1 per_tx x1 | V2 block x1000 | V3 per_tx x0.1 |
|---|---|---|---|---|---|
| proposed | 5.8 → 3.3 (-43%) | 2.8 → 2.8 (-0%) | 2.7 → 2.0 (-28%) | 2.8 → 2.8 (+3%) | 2.3 → 2.7 (+19%) |
| no_cas | 6.2 → 3.7 (-40%) | 5.5 → 5.6 (+2%) | n/a | 17.5 → 8.5 (-52%) | 35.7 → 5.2 (-85%) |
| fixed_block_size | 7.6 → 5.7 (-25%) | 1.7 → 2.1 (+29%) | 19.0 → 7.3 (-61%) | 2.3 → 1.9 (-20%) | 2.9 → 2.0 (-28%) |
| fixed_block_interval | 7.8 → 6.3 (-19%) | 4.4 → 3.3 (-24%) | 5.0 → 5.7 (+13%) | 3.9 → 3.5 (-10%) | 4.5 → 3.9 (-13%) |
| static | 8.6 → 7.2 (-16%) | 2.8 → 2.8 (+2%) | 26.1 → 12.8 (-51%) | 2.7 → 4.8 (+76%) | 5.8 → 3.9 (-32%) |
