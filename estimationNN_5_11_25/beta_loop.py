import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib.ticker import FuncFormatter
from omegaconf import DictConfig, OmegaConf
from run import main_run
from plot_helpers import to_percent

#### NOTE: REMOVED ALL code that involves br_nn or plotting_block_recursive. I assume this is no longer needed?

@hydra.main(version_base=None, config_path="config", config_name="config")
def main_plot(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))  # Print model configuration settings
    print(f"Path: {os.getcwd()}")
    
    # Aditional configuration settings
    TRAIN = True  # Flag to enable model training
    beta_values = [0.95, 0.5, 0.35, 0.15]  # Beta values to iterate over
    t_plot, dt = 500, 0.01

    # Train models if enabled
    if TRAIN:
        print("TRAINING all betas")
        cfg.train_nn.block_recursive = False # NOTE: Is this still needed?
        for beta in beta_values:
            cfg.train_nn.beta = beta
            print(f"\n\nBeta changed to {cfg.train_nn.beta}")
            main_run(cfg)  # Execute main training run with a given beta.
            
    else:
        print("Using saved models, only plotting")
    
    # Generate full dynamics plot
    cfg.train_nn.block_recursive = False # NOTE: This as well?
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    ax_titles = ["Unemployment", "Vacancies", "Poaching", "Hiring from the Unemployed"]
    formatter = FuncFormatter(to_percent)  # Format y-axis as percentage
    x_values = np.arange(t_plot) * dt  # Time axis values
    

    # Loop over beta values, and plot IRF dynamics of each in a single subplot
    for beta in beta_values:
        cfg.train_nn.beta = beta
        print(f"\n\nBeta changed to {cfg.train_nn.beta}")
        ct = instantiate(cfg.train_nn)  # Instantiate model with new beta
        
        # Load simulation data for each key
        data_keys = ["U", "V", "poach", "hire_from_u"]
        data = {key: np.load(os.path.join(ct.path, f"{key}_IRF_rec_L_ave_beta{ct.beta}.npy"))[:t_plot] for key in data_keys}
        
        # Plot each metric in the respective subplot
        for ax, key in zip(axes.flatten(), data_keys):
            ax.plot(x_values[1:], data[key][1:] / data[key][0] - 1, label=f"$\\beta={ct.beta:.2f}$")
            ax.set_xlabel("Years", fontsize=20)
            ax.set_title(ax_titles[data_keys.index(key)], fontsize=20)
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(axis="both", labelsize=20)
    
    # Add plot, title, legend, and save the figure
    fig.suptitle("1-Year Recession Dynamics, $\\eta=0.01$", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend([f"$\\beta={b:.2f}$" for b in beta_values], loc="upper center", bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=len(beta_values), fontsize=20)
    fig.savefig("IRF_rec.png", dpi=300, bbox_inches="tight")
    

if __name__ == "__main__":
    start_time = time.time()
    main_plot()
    elapsed_time = time.time() - start_time
    print(f"Plotting U IRF with different betas took {elapsed_time:.2f} seconds.")
