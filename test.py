import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from produce_data import simulate_xpcs, normalize_g2
from utils import (
    plot_g2, plot_grouped_bar, plot_bar, plot_multi_bar, plot_auto_correlation, 
    plot_multi_bar_v2, plot_nonequilibrium_measure, calc_tSNE, plot_cluster,
    calc_umap, calc_pca, nonequilibrium_measure, plot_nonequilibrium_distribution,
    plot_nonequilibrium_distribution_v2
)
from train_adv import XPCSDataset, XPCSNet
from train_vanilla import VanillaXPCSNet

if __name__ == "__main__":
    # gamma = 3e18
    # D = 1e-22
    # GB_conc = 0.1
    # T = 300.0

    # g2 = simulate_xpcs(gamma, D, GB_conc, T, seed=42)
    # torch.save(g2, "test.pt")
    
    # data_file = "test.pt"
    # g2_data = torch.load(data_file, map_location="cpu", weights_only=True).to(torch.float32).squeeze(0)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(g2, cmap='viridis', origin='lower')
    # plt.colorbar(label="$g_2(q, t_1, t_2)$")
    # plt.title("Target")
    # plt.xlabel("$t_2$")
    # plt.ylabel("$t_1$")
    # plt.savefig(f"test.pdf")
    # plt.close()
    # data = np.load("exp_data/030BM_L_dose2_T26C.npy")
    # data = torch.load("dataset/experiment/000000.pt")
    # print(data.shape)
    # df_s = pd.read_csv("dataset/simulation/manifest.csv")
    # df_s["domain"] = "simulation"
    # df_e = pd.read_csv("dataset/experiment/manifest.csv")
    # df_e["domain"] = "experiment"
    # df_s.to_csv("dataset/simulation/manifest.csv", index=False)
    # df_e.to_csv("dataset/experiment/manifest.csv", index=False)
    
    # T1, gamma1, D1, GB_conc1 = 299.1499938964844,3.337487831521034e+18,8.027100841641995e-23,0.13430529832839966 # 030BM_L_dose3_T26C
    # T2, gamma2, D2, GB_conc2 = 369.1499938964844,3.447392265054454e+18,9.804309166707584e-23,0.14749491214752197 # 030BM_L_dose3_T96C
    # T3, gamma3, D3, GB_conc3 = 466.1499938964844,3.4836376658640896e+18,1.0940549073619375e-22,0.16018903255462646 # 030BM_L_dose3_T193C
    # g2_1 = simulate_xpcs(gamma1, D1, GB_conc1, T1, seed=42)
    # plot_g2(g2_1, Path("inference_results/XPCS_best_20251114-000648/030BM_L_dose3_T193C.pdf"), f"T={T1:.1f}K")
    # g2_2 = simulate_xpcs(gamma2, D2, GB_conc2, T2, seed=42)
    # plot_g2(g2_2, Path("inference_results/XPCS_best_20251114-000648/030BM_L_dose3_T26C.pdf"), f"T={T2:.1f}K")
    # g2_3 = simulate_xpcs(gamma3, D3, GB_conc3, T3, seed=42)
    # plot_g2(g2_3, Path("inference_results/XPCS_best_20251114-000648/030BM_L_dose3_T96C.pdf"), f"T={T3:.1f}K")
    # types = (
    #     "GB stiffness\n" r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
    #     "Diffusitivity\n" r"$D$ ($10^{-23}$cm$^2$/s)",
    #     "Effective GB concentration\n" r"$\lambda_{\mathrm{GB}}$",
    # )
    # params = {
    #     r"26$^{\circ}$C": (gamma1 * 1e-18, D1 * 1e23, GB_conc1),
    #     r"96$^{\circ}$C": (gamma2 * 1e-18, D2 * 1e23, GB_conc2),
    #     r"193$^{\circ}$C": (gamma3 * 1e-18, D3 * 1e23, GB_conc3),
    # }
    # max_params = (
    #     max(gamma1, gamma2, gamma3) * 1e-18,
    #     max(D1, D2, D3) * 1e23,
    #     max(GB_conc1, GB_conc2, GB_conc3),
    # )
    # plot_grouped_bar(types, params, max_params, Path("inference_results/XPCS_best_20251114-000648/parameters.pdf"))
    # params_gamma = {
    #     r"26$^{\circ}$C": gamma1 * 1e-18,
    #     r"96$^{\circ}$C": gamma2 * 1e-18,
    #     r"193$^{\circ}$C": gamma3 * 1e-18,
    # }
    # params_D = {
    #     r"26$^{\circ}$C": D1 * 1e23,
    #     r"96$^{\circ}$C": D2 * 1e23,
    #     r"193$^{\circ}$C": D3 * 1e23,
    # }
    # params_GB = {
    #     r"26$^{\circ}$C": GB_conc1,
    #     r"96$^{\circ}$C": GB_conc2,
    #     r"193$^{\circ}$C": GB_conc3,
    # }
    
    # T4, gamma4, D4, GB_conc4 = 299.1499938964844,3.408157292128895e+18,8.39165239860635e-23,0.14071986079216003
    # T5, gamma5, D5, GB_conc5 = 369.1499938964844,3.3655484677735055e+18,9.007961024862938e-23,0.13617688417434692
    # T6, gamma6, D6, GB_conc6 = 466.1499938964844,3.425269541347787e+18,1.038483253794845e-22,0.14033278822898865
    # params_gamma_vanilla = {
    #     r"26$^{\circ}$C": gamma4 * 1e-18,
    #     r"96$^{\circ}$C": gamma5 * 1e-18,
    #     r"193$^{\circ}$C": gamma6 * 1e-18,
    # }
    # params_D_vanilla = {
    #     r"26$^{\circ}$C": D4 * 1e23,
    #     r"96$^{\circ}$C": D5 * 1e23,
    #     r"193$^{\circ}$C": D6 * 1e23,
    # }
    # params_GB_vanilla = {
    #     r"26$^{\circ}$C": GB_conc4,
    #     r"96$^{\circ}$C": GB_conc5,
    #     r"193$^{\circ}$C": GB_conc6,
    # }
    # max_params_gamma = max(gamma1, gamma2, gamma3, gamma4, gamma5, gamma6) * 1e-18
    # max_params_D = max(D1, D2, D3, D4, D5, D6) * 1e23
    # max_params_GB = max(GB_conc1, GB_conc2, GB_conc3, GB_conc4, GB_conc5, GB_conc6)
    # min_params_gamma = min(gamma1, gamma2, gamma3, gamma4, gamma5, gamma6) * 1e-18
    # min_params_D = min(D1, D2, D3, D4, D5, D6) * 1e23
    # min_params_GB = min(GB_conc1, GB_conc2, GB_conc3, GB_conc4, GB_conc5, GB_conc6)

    # plot_multi_bar(params_gamma, params_gamma_vanilla, max_params_gamma, min_params_gamma, "GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)", Path("inference_results/parameter_gamma_comparison.pdf"))
    # plot_multi_bar(params_D, params_D_vanilla, max_params_D, min_params_D, "Diffusitivity " r"$D$ ($10^{-23}$ cm$^2$/s)", Path("inference_results/parameter_D_comparison.pdf"))
    # plot_multi_bar(params_GB, params_GB_vanilla, max_params_GB, min_params_GB, "Effective GB concentration " r"$\lambda_{\mathrm{GB}}$", Path("inference_results/parameter_GB_comparison.pdf"))
    
    # plot_multi_bar_v2(params_gamma, params_gamma_vanilla, max_params_gamma, min_params_gamma, "GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)", Path("inference_results/parameter_gamma_comparison_v2.pdf"))
    # plot_multi_bar_v2(params_D, params_D_vanilla, max_params_D, min_params_D, "Diffusitivity " r"$D$ ($10^{-23}$ cm$^2$/s)", Path("inference_results/parameter_D_comparison_v2.pdf"))
    # plot_multi_bar_v2(params_GB, params_GB_vanilla, max_params_GB, min_params_GB, "Effective GB concentration " r"$\lambda_{\mathrm{GB}}$", Path("inference_results/parameter_GB_comparison_v2.pdf"))

    # plot_bar(params_gamma, max(gamma1, gamma2, gamma3) * 1e-18, "GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)", Path("inference_results/XPCS_best_20251114-000648/parameter_gamma.pdf"))
    # plot_bar(params_D, max(D1, D2, D3) * 1e23, "Diffusitivity " r"$D$ ($10^{-23}$cm$^2$/s)", Path("inference_results/XPCS_best_20251114-000648/parameter_D.pdf"))
    # plot_bar(params_GB, max(GB_conc1, GB_conc2, GB_conc3), "GB concentration " r"$\lambda_{\mathrm{GB}}$", Path("inference_results/XPCS_best_20251114-000648/parameter_GB.pdf"))
    
    # for data_file in Path("exp_data").glob("030BM_L_dose3_*.npz"):
    #     g2 = np.load(data_file)['g12']
    #     g2 = g2[:2500, :2500, 0]
    #     g2 = normalize_g2(torch.from_numpy(g2), min_val=1.0, max_val=1.2)
    #     nonequilibrium_measure_value = nonequilibrium_measure(g2)
    #     print(f"{data_file.stem}: Nonequilibrium Measure = {nonequilibrium_measure_value:.4f}")
        # plot_auto_correlation(
        #     g2, Path(f"inference_results/XPCS_best_20251114-000648/{data_file.stem}_auto_correlation.pdf"),
        # )
        # plot_nonequilibrium_measure(
        #     torch.from_numpy(g2), 
        #     Path(f"inference_results/XPCS_best_20251114-000648/{data_file.stem}_nonequilibrium_measure.pdf"),
        #     xticks=[0, 1250, 2500], yticks=[1250, 2500],
        # )
    
    # data = torch.load("dataset/simulation/000628.pt", weights_only=True).squeeze(0)
    # data = normalize_g2(data, min_val=1.0, max_val=1.2)
    # plot_g2(
    #     data, Path("inference_results/XPCS_best_20251114-000648/simulated.pdf"),
    #     xlabel="Time $t_2$ (s)", ylabel="Time $t_1$ (s)",
    #     xticks=[0, 128, 256], xlabels=[0, 1250, 2500],
    #     yticks=[128, 256], ylabels=[1250, 2500],
    #     colorbar=True,
    # )
    
    # sim_dataset = XPCSDataset(Path("dataset/simulation"))
    # exp_dataset = XPCSDataset(Path("dataset/experiment"))
    # model = XPCSNet()
    # model.load_state_dict(
    #     torch.load(
    #         "models/XPCS_best_20251114-000648.pt",
    #         weights_only=True,
    #     )
    # )
    # model_v = VanillaXPCSNet()
    # model_v.load_state_dict(
    #     torch.load(
    #         "models/Vanilla_XPCS_best_20251114-003841.pt",
    #         weights_only=True,
    #     )
    # )
    # X_tsne, domain_labels = calc_tSNE(model, sim_dataset, exp_dataset, max_iter=900, init='random')
    # X_tsne_V, domain_labels_V = calc_tSNE(model_v, sim_dataset, exp_dataset, max_iter=900, init='random')
    # plot_cluster(
    #     X_tsne,
    #     domain_labels,
    #     Path("inference_results/XPCS_best_20251114-000648/tSNE.pdf"),
    # )
    # plot_cluster(
    #     X_tsne_V,
    #     domain_labels_V,
    #     Path("inference_results/XPCS_best_20251114-000648/tSNE_vanilla.pdf"),
    # )
    # X_umap, domain_labels = calc_umap(model, sim_dataset, exp_dataset, n_neighbors=5, min_dist=0.1, init='random')
    # X_umap_V, domain_labels_V = calc_umap(model_v, sim_dataset, exp_dataset, init='pca')
    # # save UMAP coordinates for future use
    # with open("inference_results/sim_exp_umap_coordinates.npz", "wb") as f:
    #     np.savez(
    #         f,
    #         X_umap=X_umap,
    #         domain_labels=domain_labels,
    #         X_umap_V=X_umap_V,
    #         domain_labels_V=domain_labels_V,
    #     )
    # Read UMAP coordinates from file
    # with open("inference_results/sim_exp_umap_coordinates.npz", "rb") as f:
    #     data = np.load(f)
    #     X_umap = data['X_umap']
    #     domain_labels = data['domain_labels']
    #     X_umap_V = data['X_umap_V']
    #     domain_labels_V = data['domain_labels_V']
    
    # indices = np.random.choice(len(X_umap), size=200, replace=False)
    # X_umap = X_umap[indices]
    # domain_labels = domain_labels[indices]
    # X_umap_V = X_umap_V[indices]
    # domain_labels_V = domain_labels_V[indices]
    
    # plot_cluster(
    #     X_umap,
    #     domain_labels,
    #     Path("inference_results/UMAP(2).pdf"),
    #     sim_marker='o',
    #     exp_marker='o',
    #     sim_marker_size=600,
    #     exp_marker_size=600,
    # )
    # plot_cluster(
    #     X_umap_V,
    #     domain_labels_V,
    #     Path("inference_results/UMAP_vanilla(2).pdf"),
    #     sim_marker='s',
    #     exp_marker='^',
    #     sim_marker_size=600,
    #     exp_marker_size=600,
    # )
    # X_pca, domain_labels = calc_pca(model, sim_dataset, exp_dataset)
    # X_pca_V, domain_labels_V = calc_pca(model_v, sim_dataset, exp_dataset)
    # plot_cluster(
    #     X_pca,
    #     domain_labels,
    #     Path("inference_results/PCA.pdf"),
    # )
    # plot_cluster(
    #     X_pca_V,
    #     domain_labels_V,
    #     Path("inference_results/PCA_vanilla.pdf"),
    # )
    
    sim_path = Path("dataset/simulation")
    with open(sim_path / "manifest_with_non_equ_1.csv", "r") as f:
        df = pd.read_csv(f)
        
    # nonequilibrium_measures = []
    # for _, row in tqdm(df.iterrows()):
    #     g2 = torch.load(row["path"], weights_only=True).squeeze(0)
    #     g2 = normalize_g2(g2, min_val=1.0, max_val=1.2)
    #     measure = nonequilibrium_measure(g2)
    #     nonequilibrium_measures.append(measure)

    # df['nonequilibrium_measure'] = nonequilibrium_measures
    nonequilibrium_measures = df["nonequilibrium_measure"]
    # df.to_csv(sim_path / "manifest_with_non_equ_1.csv", index=False)
    
    # # a histogram of nonequilibrium measures
    # plt.figure(figsize=(10, 6))
    # plt.hist(nonequilibrium_measures, bins=50, color='blue', alpha=0.7)
    # plt.savefig(Path("inference_results/nonequilibrium_measure_histogram.pdf"))
    
    df.sort_values(by='nonequilibrium_measure', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Extract top 4 quartiles
    idxs = [362, 805, 1279, 1608]
    quartile_size = len(df) // 4
    quartiles = [
        df.iloc[:quartile_size],
        df.iloc[quartile_size:2*quartile_size],
        df.iloc[2*quartile_size:3*quartile_size],
        df.iloc[3*quartile_size:],
    ]
    save_path = Path("inference_results") / "Simulation"
    save_path.mkdir(parents=True, exist_ok=True)
    for i, quartile in enumerate(quartiles):
        # idx = np.random.choice(quartile.index)
        if i > 0: break
        idx = idxs[i]
        row = quartile.loc[idx]
        print(idx)  # document which sample is used
        g2 = torch.load(row["path"], weights_only=True).squeeze(0)
        g2 = normalize_g2(g2, min_val=1.0, max_val=1.2)
        # plot_nonequilibrium_measure(
        #     g2,
        #     save_path / f"nonequilibrium_measure_quartile_{i+1}(1).pdf",
        #     xticks=[0, 128, 256],
        #     xlabels=[0, 1250, 2500],
        #     yticks=[128, 256],
        #     ylabels=[1250, 2500],
        # )
        # plot_g2(
        #     g2,
        #     save_path / f"g2_quartile_{i+1}(1).pdf",
        #     xlabel="Time $t_2$ (s)", ylabel="Time $t_1$ (s)",
        #     xticks=[0, 128, 256], xlabels=[0, 1250, 2500],
        #     yticks=[128, 256], ylabels=[1250, 2500],
        #     colorbar=True,
        # )
        plot_auto_correlation(
            g2.to(torch.float32),
            save_path / f"auto_correlation_quartile_{i+1}(1).pdf",
            xticks=[0, 128, 256],
            xlabels=[0, 1250, 2500],
            yticks=[1.0, 1.1, 1.2],
            smooth=True,
            legend=True,
        )
    # idxs = [362, 805, 1279, 1608]
    # for idx in idxs:
    #     row = df.loc[idx]
    #     gamma, D, GB_conc, T = row["gamma"], row["D"], row["GB_conc"], row["T"]
    #     nonequilibrium_measure_value = row["nonequilibrium_measure"]
    #     print(f"Index: {idx}, Gamma: {gamma}, D: {D}, GB_conc: {GB_conc}, T: {T}, Nonequilibrium Measure: {nonequilibrium_measure_value}")
    
    # sim_path = Path("dataset/simulation")
    # with open(sim_path / "manifest_with_non_equ_1.csv", "r") as f:
    #     df = pd.read_csv(f)
        
    # plot_nonequilibrium_distribution_v2(
    #     df,
    #     Path("inference_results/non_equ_dist_D_gamma(1).pdf"),
    #     x="D",
    #     y="gamma",
    #     xscale="log",
    #     yscale="linear",
    #     xlabel="Diffusitivity " r"$D$ (cm$^2$ $\cdot$s$^{-1}$)",
    #     ylabel="GB stiffness " r"$\Gamma$ ($10^{18}$ s$\cdot$cm$^{-2}$)",
    #     xticks=[1e-23, 1e-22, 1e-21],
    #     yticks=[2e18, 2.5e18, 3e18, 3.5e18, 4e18, 4.5e18, 5e18],
    #     xlabels=["$10^{-23}$", "$10^{-22}$", "$10^{-21}$"],
    #     ylabels=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    # )
        
    # plot_nonequilibrium_distribution_v2(
    #     df,
    #     Path("inference_results/non_equ_dist_gamma_GB(1).pdf"),
    #     x="gamma",
    #     y="GB_conc",
    #     xscale="linear",
    #     yscale="linear",
    #     xlabel="GB stiffness " r"$\Gamma$ (s$\cdot$cm$^{-2}$)",
    #     ylabel="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
    #     xticks=[2e18, 3e18, 4e18, 5e18],
    #     yticks=[0.0, 0.1, 0.2, 0.3],
    #     xlabels=[2, 3, 4, 5],
    #     ylabels=[0.0, 0.1, 0.2, 0.3],
    # )
    
    # plot_nonequilibrium_distribution_v2(
    #     df,
    #     Path("inference_results/non_equ_dist_D_GB(1).pdf"),
    #     x="D",
    #     y="GB_conc",
    #     xscale="log",
    #     yscale="linear",
    #     xlabel="Diffusitivity " r"$D$ (cm$^2$ $\cdot$s$^{-1}$)",
    #     ylabel="Effective GB concentration " r"$\lambda_{\mathrm{GB}}$",
    #     xticks=[1e-23, 1e-22, 1e-21],
    #     yticks=[0.00, 0.10, 0.20, 0.30],
    #     xlabels=["$10^{-23}$", "$10^{-22}$", "$10^{-21}$"],
    #     ylabels=[0.00, 0.10, 0.20, 0.30],
    # )
    