# Training Logs

## Adversarial Training

1. `XPCS_best_20251114-000648.pt` (GB_Train_21135):
- Dynamic adversarial training with learning rate 3e-4
- Epoch: 6/100, Train Loss: 0.054580 (Pred: 0.068222, Class: 0.037120), Val Loss: 0.059049 (Pred: 0.066365, Class: 0.044724); Per-parameter MAE: gamma: 7.8891e+17, D: 9.9438e-23, GB_conc: 7.5341e-02
- Test MAE [gamma]: 7.977e+17
- Test MAE [D]: 8.669e-23
- Test MAE [GB_conc]: 7.333e-02
- Example of `030BM_L_dose2`:

| T | gamma | D | GB_conc |
|-----|-------|-------|---------|
| 26 | 3.145e+18 | 3.132e-23 | 0.103 |
| 96 | 3.445e+18 | 3.617e-23 | 0.124 |
| 193 | 3.870e+18 | 4.599e-23 | 0.154 |


## Vanilla Training

1. `Vanilla_XPCS_best_20251114-003841.pt` (GB_Train_21136):

- Test MAE [gamma]: 7.486e+17
- Test MAE [D]: 2.071e-23
- Test MAE [GB_conc]: 2.270e-02




