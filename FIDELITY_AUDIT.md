# FIDELITY_AUDIT

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | HR teacher pretrain | 85% | CelebA proxy |
| 2 | Shared Softmax weights | 95% | Tied classifier |
| 3 | KL + CE on HR/LR student | 92% | Synthetic LR |
| 4 | Multi-scale LFW + figures | 93% | D2 |
| 5 | Size gate + publish | 95% | Stop |

Confidence vs paper structure: **~95%**.

## Measured run (full)

- CelebA+LFW 3.09 GiB; teacher 8 + student 12 epochs; 1658.4 s
- LFW ×1.0: teacher 58.97% / student 58.05%
- LFW ×0.5: teacher 59.67% / student 58.10%
- LFW ×0.25: teacher 59.85% / student 58.17%
