import os
import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from train_nn import Train_NN, Master_PINN_S
from calibrator import Calibrator

@hydra.main(version_base=None, config_path="config", config_name="config")
def main_calibration(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed) # Set seed for reproducibility
    print(OmegaConf.to_yaml(cfg)) # Print model configuration settings
    
    # Initialize Training and Calibration objects
    ct: Train_NN = instantiate(cfg.train_nn)
    ct.init_torch()
    cb: Calibrator = instantiate(cfg.calibrator)
    
    # Device (ideally) should be 'cuda'
    device = ct.device
    print(f"Device: {device}")
    
    # Load Surplus PINN Model
    pinn_S = Master_PINN_S(
        nn_width=ct.nn_width,
        nn_num_layers=ct.nn_num_layers,
        n_x=ct.nx,
        n_y=ct.ny,
        n_par=len(ct.par)
    ).to(device)
    pinn_S = torch.nn.DataParallel(pinn_S).float()
    
    model_path = os.path.join(ct.path, "model_S_final.pt")
    pinn_S.load_state_dict(torch.load(model_path, map_location=device)['modelS_state_dict'])
    
    # Load Value function PINN models if wage_dynamics is enabled
    pinn_V_u, pinn_V_v = None, None
    if ct.wage_dynamics:
        pinn_V_u = Master_PINN_V_u(
            nn_width=ct.nn_width,
            nn_num_layers=ct.nn_num_layers,
            n_x=ct.nx,
            n_y=ct.ny,
            n_par=len(ct.par)
        ).to(device)
        pinn_V_u = torch.nn.DataParallel(pinn_V_u).float()
        pinn_V_u.load_state_dict(torch.load(os.path.join(ct.path, "model_V_u.pt"), map_location=device)['modelV_u_state_dict'])
        
        pinn_V_v = Master_PINN_V_v(
            nn_width=ct.nn_width,
            nn_num_layers=ct.nn_num_layers,
            n_x=ct.nx,
            n_y=ct.ny,
            n_par=len(ct.par)
        ).to(device)
        pinn_V_v = torch.nn.DataParallel(pinn_V_v).float()
        pinn_V_v.load_state_dict(torch.load(os.path.join(ct.path, "model_V_v.pt"), map_location=device)['modelV_v_state_dict'])
    
    # Perform Calibration
    cb.set_instances(ct, pinn_S, cfg, pinn_V_u, pinn_V_v)
    cb.calibrate()
    print(f"Working directory: {os.path.join(ct.working_dir, ct.path)}")

if __name__ == "__main__":
    start_time = time.time()
    main_calibration()
    elapsed_time = time.time() - start_time
    print(f"Program completed in {elapsed_time:.2f} seconds.")
