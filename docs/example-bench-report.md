# Benchmark — 2026-09-10T00:41:45+08:00

- endpoint: `http://127.0.0.1:8080`
- model: `Qwen3.8-27B-Q6-dual-5060ti`
- gpus:
  - 0, NVIDIA GeForce RTX 5060 Ti, 15517 MiB, 16311 MiB
  - 1, NVIDIA GeForce RTX 5060 Ti, 15517 MiB, 16311 MiB

## 1. Single-stream decode / 单流解码
```
model: Qwen3.8-27B-Q6-dual-5060ti
thinking-off: prompt=49 completion=512 wall=4.32s decode=118.6 tok/s
thinking-on: prompt=89 completion=512 wall=6.70s decode=76.4 tok/s
```

## 2. Concurrency (shared paged KV pool) / 并发（共享 KV 池）

```
C=1: wall=   3.1s total_completion=  256 aggregate=  81.7 tok/s  per-stream=[81.7]
C=2: wall=   3.4s total_completion=  512 aggregate= 150.3 tok/s  per-stream=[76.7, 75.1]
C=4: wall=   4.0s total_completion= 1024 aggregate= 256.8 tok/s  per-stream=[69.2, 69.3, 67.9, 64.2]
C=8: wall=   7.7s total_completion= 2048 aggregate= 264.9 tok/s  per-stream=[69.3, 69.3, 66.3, 33.7, 68.6, 33.7, 33.1, 33.7]
```

## 3. Long-context needle / 长文召回

```
needle target=100000 prompt_tokens=96588 wall=69.2s prefill=1396 tok/s recall 5/5: ['ZEBRA-ALPHA-8842', 'ZEBRA-BETA-3391', 'ZEBRA-GAMMA-7705', 'ZEBRA-DELTA-6617', 'ZEBRA-OMEGA-2054']
```

