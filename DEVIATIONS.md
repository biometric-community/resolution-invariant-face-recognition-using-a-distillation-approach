# DEVIATIONS — Resolution-Invariant Distillation

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train corpus | FaceScrub / MegaFace HR | CelebA identities | MegaFace empty locally |
| D2 | LR benchmarks | TinyFace, SCFace | LFW downsampled ×1/×0.5/×0.25 | TinyFace/SCFace unavailable |
| D3 | Backbone | Paper FR CNN | ResNet-18 + shared Softmax | Same distillation recipe |
| D4 | Epochs | Paper schedule | Teacher 8 + Student 12 | Practical full-train budget |
