import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import time
import torch
import os
import torch.optim as optim
from train_nn import Train_NN, Master_PINN_S
from plot import plot_loss, plot_ergodic_g_S_alpha
from simulate_class import Simulation
from copy import deepcopy

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    main_run(cfg)

def main_run(cfg: DictConfig):
    # Print the configuration for reference
    print(OmegaConf.to_yaml(cfg))

    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed)
    
    # Instantiate the Train_NN class using configuration
    ct: Train_NN = instantiate(cfg.train_nn)

    # Solve steady state and generate initial guess for U and V
    ct.solve_steady_state(ct.par, True)
    ct.init_torch()
    ct.init_g_sample_space()

    # Device should (ideally) be cuda/GPU
    print('Device: ', ct.device)

    # Instantiate the calibrator using configuration
    cb = instantiate(cfg.calibrator)

    # Initialize the Master_PINN_S model for Surplus value function
    pinn_S = Master_PINN_S(
        nn_width=ct.nn_width,
        nn_num_layers=ct.nn_num_layers,
        n_x=ct.nx,
        n_y=ct.ny,
        n_par=len(ct.par)
    ).to(ct.device)

    # Wrap the model in DataParallel for multi-GPU support
    pinn_S = torch.nn.DataParallel(pinn_S)
    pinn_S = pinn_S.float()

    # No wage dynamics
    pinn_V_u = None
    pinn_V_v = None

    # Check if loading a saved model for Surplus
    if not ct.loading_saved_model_S:
        print('New NN for S')
        checkpoint_path = os.path.join(ct.path, "model_S_init.pt") 

        # Load pre-trained model if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            pinn_S.load_state_dict(checkpoint['modelS_state_dict'])

        # Pretrain the Surplus NN to match the initial guess, if not loading a saved model
        if not ct.loading_saved_init_S:
            print("Starting surplus NN training on initial guess")
            pre_training_optimizer_S = optim.Adam(pinn_S.parameters(), lr=ct.lr_init)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                pre_training_optimizer_S, T_0=10, T_mult=1, eta_min=0,
                last_epoch=-1)
            ct.initial_guess(pinn_S, pre_training_optimizer_S, scheduler,
                                epochs=ct.epochs_init)
            print("Finished surplus NN training on initial guess")
            torch.save({
                'modelS_state_dict': pinn_S.state_dict(),
                'optimizer_state_dict': pre_training_optimizer_S.state_dict(),
            }, os.path.join(ct.path, "model_S_init.pt"))
        else:
            print("Loaded existing pretrain NN model for S, no further training on initial guess")


        # Main training for Surplus NN to solve the Master equation
        # First training phase
        training_optimizer_S = optim.Adam(pinn_S.parameters(), lr=ct.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            training_optimizer_S, T_0=10, T_mult=1, eta_min=0,
            last_epoch=-1)
        print(f"Starting main training to solve surplus PDE,lr={ct.lr}")
        ct.training(pinn_S, pinn_S, training_optimizer_S, scheduler,
                    epochs=ct.epochs, save_path=ct.path, save_freq=ct.save_freq,
                    option='S', N=ct.sample_low)

        # Second training phase: lower learning rate and increased # of sampled points for refinement
        print(f"Starting main training with lower learning rate={ct.lr_low:.4e}")
        training_optimizer_S = optim.Adam(pinn_S.parameters(), lr=ct.lr_low)
        ct.training(pinn_S, pinn_S, training_optimizer_S, scheduler,
                    epochs=ct.epochs_low, save_path=ct.path, save_freq=ct.save_freq,
                    option='S', N=ct.sample_high, rewrite_loss_file=False)
        
        # Save the final trained Surplus model
        torch.save({
            'modelS_state_dict': pinn_S.state_dict(),
            'optimizer_state_dict': training_optimizer_S.state_dict(),
        }, os.path.join(ct.path, "model_S_final.pt"))

        # Plot surplus loss during training
        plot_loss(ct)
        
    else:
        print('Saved S model')
        checkpoint = torch.load(os.path.join(ct.path, "model_S_final.pt"), map_location=ct.device)
        pinn_S.load_state_dict(checkpoint['modelS_state_dict'])


    # Calibration
    cb = instantiate(cfg.calibrator)
    cb.set_instances(ct, pinn_S, cfg, pinn_V_u, pinn_V_v)
    cb.calibrate()
    cb.load_calibration_result()
    ct.par.update(cb.optimal_par)

    # need to solve ss first?
    #ct.load_g_ss = False
    #ct.solve_steady_state(ct.par, True)

    #sim = Simulation(ct, pinn_S, ct.par, cfg=cfg)
    #sim.ergodics(0.01, 5001, 10000, False, printer=True)

    # Plot ergodic, g, S, alpha at optimal calibration parameters
    plot_ergodic_g_S_alpha(ct)
          
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f'Program completed after {end - start:.2f} seconds.')