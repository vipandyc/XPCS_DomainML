# Contrastive diagnostics

Batch Job | Result Dir | Training config | results
--- | --- | --- | ---
21855 | coral_noT_top-left_20260420-185415 | contrastive_weight=1.0, bandwidth=0.4, margin=0.5 | UMAP good, sim good, contrastive bad
21854 | coral_noT_top-left_20260420-185413 | contrastive_weight=1.0, bandwidth=0.3, margin=0.5 | UMAP good, sim good, contrastive bad
21850 | coral_noT_top-left_20260420-075408 | contrastive_weight=1.0, bandwidth=1.0, margin=0.5 | UMAP good, sim not good, contrastive equally bad
21849 | coral_noT_top-left_20260420-090904 | contrastive_weight=1.0, bandwidth=0.5, margin=0.5 | UMAP good, sim not good, contrastive equally bad
21848 | coral_noT_top-left_20260420-090634 | contrastive_weight=4.0, bandwidth=0.2, margin=0.5 | UMAP good, sim not good, contrastive collapsed
21847 | coral_noT_top-left_20260420-081546 | contrastive_weight=2.0, bandwidth=0.2, margin=0.5 | UMAP good, sim not good, contrastive collapsed 
21840 | coral_noT_top-left_20260419-004712 | contrastive_weight=1.0, bandwidth=0.05, margin=0.3 | UMAP good, sim good, contrastive bad, model won't predict low D & high lambda (same as 21838)
21839 | coral_noT_top-left_20260419-004645 | contrastive_weight=10.0, bandwidth=0.2, margin=0.5 | UMAP good, sim completely collapsed
21838 | coral_noT_top-left_20260419-003318 | contrastive_weight=1.0, bandwidth=0.2, margin=0.5 | UMAP ok, sim good, contrastive bad


