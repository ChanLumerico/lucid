# Lucid Performance Baseline

**Date:** 2026-04-29  
**Backends:** lucid-cpu, lucid-gpu, torch-cpu, torch-mps  
**Iterations:** 30 (warm-up: 8)  
**`% of torch`** = lucid-cpu latency ÷ torch-cpu latency × 100 (100% = same speed, <100% = faster)

## Matmul
  matmul [128,128]x[128,128]... [lucid-cpu: 6.1 µs] [lucid-gpu: 24.7 µs] [torch-cpu: 5.8 µs] [torch-mps: 264.6 µs] 
  matmul [512,512]x[512,512]... [lucid-cpu: 157.1 µs] [lucid-gpu: 37.5 µs] [torch-cpu: 144.4 µs] [torch-mps: 623.9 µs] 
  matmul [1024,1024]x[1024,1024]... [lucid-cpu: 1.64 ms] [lucid-gpu: 27.5 µs] [torch-cpu: 1.04 ms] [torch-mps: 858.2 µs] 
  matmul [2048,2048]x[2048,2048]... [lucid-cpu: 14.30 ms] [lucid-gpu: 22.0 µs] [torch-cpu: 14.23 ms] [torch-mps: 5.52 ms] 
  matmul [64,4096]x[4096,4096]... [lucid-cpu: 7.12 ms] [lucid-gpu: 26.5 µs] [torch-cpu: 7.13 ms] [torch-mps: 1.29 ms] 

| Op | lucid-cpu | lucid-gpu | torch-cpu | torch-mps | GFLOPs | % of torch |
| --- | --- | --- | --- | --- | --- | --- |
| matmul [128,128]x[128,128] | 6.1 µs | 24.7 µs | 5.8 µs | 264.6 µs | 691.9 | 104% |
| matmul [512,512]x[512,512] | 157.1 µs | 37.5 µs | 144.4 µs | 623.9 µs | 1709.1 | 109% |
| matmul [1024,1024]x[1024,1024] | 1.64 ms | 27.5 µs | 1.04 ms | 858.2 µs | 1308.9 | 157% |
| matmul [2048,2048]x[2048,2048] | 14.30 ms | 22.0 µs | 14.23 ms | 5.52 ms | 1201.4 | 101% |
| matmul [64,4096]x[4096,4096] | 7.12 ms | 26.5 µs | 7.13 ms | 1.29 ms | 301.6 | 100% |

## Conv2d
  conv2d [1,3,224,224] → 64×7×7 s2p3... [lucid-cpu: 1.82 ms] [lucid-gpu: 31.3 µs] [torch-cpu: 1.12 ms] [torch-mps: 1.10 ms] 
  conv2d [4,64,56,56] → 64×3×3 s1p1... [lucid-cpu: 5.42 ms] [lucid-gpu: 26.8 µs] [torch-cpu: 1.75 ms] [torch-mps: 747.3 µs] 
  conv2d [4,256,28,28] → 512×3×3 s1p1... [lucid-cpu: 9.98 ms] [lucid-gpu: 28.0 µs] [torch-cpu: 6.83 ms] [torch-mps: 2.31 ms] 
  conv2d [16,3,32,32] → 64×3×3 s1p1... [lucid-cpu: 539.9 µs] [lucid-gpu: 25.8 µs] [torch-cpu: 1.81 ms] [torch-mps: 520.1 µs] 

| Op | lucid-cpu | lucid-gpu | torch-cpu | torch-mps | GFLOPs | % of torch |
| --- | --- | --- | --- | --- | --- | --- |
| conv2d [1,3,224,224] → 64×7×7 s2p3 | 1.82 ms | 31.3 µs | 1.12 ms | 1.10 ms | 130.0 | 162% |
| conv2d [4,64,56,56] → 64×3×3 s1p1 | 5.42 ms | 26.8 µs | 1.75 ms | 747.3 µs | 170.5 | 310% |
| conv2d [4,256,28,28] → 512×3×3 s1p1 | 9.98 ms | 28.0 µs | 6.83 ms | 2.31 ms | 741.7 | 146% |
| conv2d [16,3,32,32] → 64×3×3 s1p1 | 539.9 µs | 25.8 µs | 1.81 ms | 520.1 µs | 104.9 | 30% |

## Softmax + SDPA
  softmax [4,512,512]... [lucid-cpu: 780.3 µs] [lucid-gpu: 23.3 µs] [torch-cpu: 371.8 µs] [torch-mps: 701.7 µs] 
  softmax [32,128,1024]... [lucid-cpu: 2.91 ms] [lucid-gpu: 38.2 µs] [torch-cpu: 1.85 ms] [torch-mps: 602.0 µs] 
  sdpa [2,8,64,64]... [lucid-cpu: 96.6 µs] [lucid-gpu: 28.4 µs] [torch-cpu: 92.3 µs] [torch-mps: 687.5 µs] 
  sdpa [2,8,128,64]... [lucid-cpu: 324.0 µs] [lucid-gpu: 26.5 µs] [torch-cpu: 285.0 µs] [torch-mps: 784.1 µs] 
  sdpa [2,16,256,64]... [lucid-cpu: 2.77 ms] [lucid-gpu: 33.6 µs] [torch-cpu: 1.18 ms] [torch-mps: 1.44 ms] 

| Op | lucid-cpu | lucid-gpu | torch-cpu | torch-mps | GFLOPs | % of torch |
| --- | --- | --- | --- | --- | --- | --- |
| softmax [4,512,512] | 780.3 µs | 23.3 µs | 371.8 µs | 701.7 µs | 6.7 | 210% |
| softmax [32,128,1024] | 2.91 ms | 38.2 µs | 1.85 ms | 602.0 µs | 7.2 | 158% |
| sdpa [2,8,64,64] | 96.6 µs | 28.4 µs | 92.3 µs | 687.5 µs | 173.7 | 105% |
| sdpa [2,8,128,64] | 324.0 µs | 26.5 µs | 285.0 µs | 784.1 µs | 207.1 | 114% |
| sdpa [2,16,256,64] | 2.77 ms | 33.6 µs | 1.18 ms | 1.44 ms | 193.5 | 235% |

## Norm
  batch_norm [4,64,56,56]... [lucid-cpu: 201.9 µs] [lucid-gpu: 35.3 µs] [torch-cpu: 521.5 µs] [torch-mps: 655.6 µs] 
  batch_norm [4,256,28,28]... [lucid-cpu: 220.0 µs] [lucid-gpu: 33.8 µs] [torch-cpu: 561.7 µs] [torch-mps: 658.7 µs] 
  batch_norm [4,512,14,14]... [lucid-cpu: 110.9 µs] [lucid-gpu: 48.9 µs] [torch-cpu: 314.4 µs] [torch-mps: 524.8 µs] 
  layer_norm [16,64,512]... [lucid-cpu: 200.1 µs] [lucid-gpu: 33.4 µs] [torch-cpu: 146.4 µs] [torch-mps: 357.0 µs] 
  layer_norm [4,512,1024]... [lucid-cpu: 777.5 µs] [lucid-gpu: 33.7 µs] [torch-cpu: 333.2 µs] [torch-mps: 592.9 µs] 

| Op | lucid-cpu | lucid-gpu | torch-cpu | torch-mps | GFLOPs | % of torch |
| --- | --- | --- | --- | --- | --- | --- |
| batch_norm [4,64,56,56] | 201.9 µs | 35.3 µs | 521.5 µs | 655.6 µs | 15.9 | 39% |
| batch_norm [4,256,28,28] | 220.0 µs | 33.8 µs | 561.7 µs | 658.7 µs | 14.6 | 39% |
| batch_norm [4,512,14,14] | 110.9 µs | 48.9 µs | 314.4 µs | 524.8 µs | 14.5 | 35% |
| layer_norm [16,64,512] | 200.1 µs | 33.4 µs | 146.4 µs | 357.0 µs | 13.1 | 137% |
| layer_norm [4,512,1024] | 777.5 µs | 33.7 µs | 333.2 µs | 592.9 µs | 13.5 | 233% |

