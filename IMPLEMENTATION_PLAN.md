# Implementation Plan

1. Train HR teacher with CE on CelebA
2. Freeze teacher; share Softmax with student
3. Train student on random-resolution (HR/LR) batches with CE + KL
4. Eval LFW at scales 1.0 / 0.5 / 0.25 for teacher vs student
