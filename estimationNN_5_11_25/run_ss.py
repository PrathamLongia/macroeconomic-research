# -*- coding: utf-8 -*-
# @author: Runhuan Wang
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    main_run(cfg)


def main_run(cfg: DictConfig):
    import time
    import torch
    import os
    from train_nn import  Train_NN
    from ss_surrogate import Steady_State_Surrogate
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    start = time.time()

    ct: Train_NN = instantiate(cfg.train_nn)
    ct.load_g_ss = False
    ct.solve_steady_state(ct.par, True, True)  # initial guess & plot & txt of U,V

    end = time.time()
    print(f'Program with nx = {ct.nx}, ny = {ct.ny} takes {end - start:.2f} seconds.\n', ct.working_dir + '\\' + ct.path)


if __name__ == "__main__":
    main()
