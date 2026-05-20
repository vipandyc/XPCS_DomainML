Experiment | Configs | Result
---|---|---
torch_mlp_2026-04-12_23-51-50_umap | LeakyReLU(0.01), 128->32->2, No weight decay | ✅
torch_mlp_2026-04-12_23-58-16_umap | LeakyReLU(0.01), 128->64->32->2, No weight decay | ✅
torch_mlp_2026-04-13_00-00-18_umap | LeakyReLU(0.01), 128->128->64->2, No weight decay | ✅
torch_mlp_2026-04-13_00-06-35_umap | ReLU, 128->128->64->2, No weight decay | ✅
torch_mlp_2026-04-13_00-08-26_umap | ReLU, 128->128->64->2, No weight decay, Un-standardized | ❌
torch_mlp_2026-04-13_00-17-49_umap | ReLU, 128->64->32->2, No weight decay, Un-standardized | ❌
torch_mlp_2026-04-13_00-19-38_umap | ReLU, 128->32->2, No weight decay, Un-standardized | ❌
