"""
filename: env.py
@authors: Jonathan Payne, Adam Rebei, Yucheng Yang

This file contains the economic environment parent class and functions
for solving the OJS model.

"""

import os
import numpy as np
import torch
from structures import Parameters, Ranges, Element
from typing import Dict, Union
from omegaconf import DictConfig
from helpers import Saver, format_params

class Environment():
    """
    Parent environment class.
    """

    def __init__(self, **kwargs):
        self._par = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
        # add estimable parameters with different model specifications

        self.init_var()

        self._par = Parameters(self.par, callback=self._update_attribute)  # callback function is for updating attributes when par is updated
        self.range = Ranges(self.par, self.par_range, device='cpu')  # include calibrated params--those don't need estimation
        self.par_draw = None

        self.init_path()

        self.saver = Saver(self.path)  # to save tensor&ndarray in the subfolder

    def _update_attribute(self, key, value):
        """callback function to update attributes when par is updated"""
        setattr(self, key, value)
        if hasattr(self, 'range'):
            Ranges.__setattr__(self.range, key, value)


    @property
    def par(self):
        return self._par

    @par.setter
    def par(self, dict_value):
        if isinstance(dict_value, (dict, DictConfig)):
            for key, value in dict_value.items():
                self._par[key] = value  # __setitem__ of Parameters will be called, which will call _update_attribute
        elif isinstance(dict_value, Parameters):
            self._par = dict_value  # update directly
        else:
            raise ValueError(f"'par' must be set to a dict|DictConfig|Parameters, not {type(dict_value)}")

    def __setattr__(self, key, value):
        """Iff an attribute is in self.par and is of right types, whenever this attribute is updated, corresponding self.par[key] is also updated """
        if key == '_par':  # if this attribute is self._par, then return immediately to avoid recursion
            super().__setattr__(key, value)
            return
        elif key == 'par':
            pass  # @par.setter will automatically update self.par and self._par
        else:
            if hasattr(self, 'par'):  # some attributes are set before self.par is created
                if key in self._par:  # if this attribute is in self.par, update self.par[key] accordingly
                    if isinstance(value, (int, float, np.ndarray, torch.Tensor))&isinstance(self._par, Parameters):
                        object.__setattr__(self._par, key, value)
                        if hasattr(self, 'range'):
                            Ranges.__setattr__(self.range, key, value)
                    elif isinstance(self._par, dict):
                        pass  # self._par hasn't been converted to Parameters, and cannot be updated directly
                    else:
                        raise TypeError(f'Invalid _par {key} value type:{type(key)}')

        super().__setattr__(key, value)  # need to create this attribute no matter adding to self._par or not

    # def __getattr__(self, name):
    #     return self._par.get(name)
    
    def __setitem__(self, key, value):
        """obj['key']=..."""
        setattr(self, key, value)  # should automatically update self.par too

    def update(self, dict: Dict):
        """obj.update({key:value})"""
        for key, value in dict.items():
            setattr(self, key, value)  # should automatically update self.par too

    def add_par(self, *args):
        """add key(&value) to self.par
        If an attribute already exist, set self.par['attribute name']=attribute value, i.e. does NOT update even input Dict
        Use: add_to_par_list("existent_attr", "new_key", {"new_attr": value})"""
        if not hasattr(self, 'par'):
            self.par = {}
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    if hasattr(self, key):
                        self.par[key] = getattr(self, key)  # NO update
                        print(f'Attribute {key}={getattr(self, key)} ALREADY exist, NkOT updating to {value}')
                    else:
                        self.par[key] = value  # update
                        setattr(self, key, value)
            elif isinstance(arg, str):
                if hasattr(self, arg):
                    self.par[arg] = getattr(self, arg)
                else:
                    self.par[arg] = None  # default value=None
            elif isinstance(arg, list):
                for item in arg:
                    if hasattr(self, item):
                        self.par[item] = getattr(self, item)  # no update
                        # print(f'Attribute {item}={getattr(self, item)} ALREADY exist, NOT updating}')
                    else:
                        self.par[item] = None
            else:
                raise ValueError(f"Unsupported par type: {type(arg)}")

    @property
    def lams(self):
        if isinstance(self.lam_L, (int, float, np.ndarray)) and isinstance(self.lam_H, (int, float, np.ndarray)):   
            return torch.tensor([self.lam_L, self.lam_H]).to(self.device)
        elif isinstance(self.lam_L, torch.Tensor) and isinstance(self.lam_H, torch.Tensor):
            return torch.stack([self.lam_L, self.lam_H])
        else:
            raise ValueError(f"Invalid lam_L and lam_H types: {type(self.lam_L)}, {type(self.lam_H)}")


    @property
    def z_L(self):
        return self.z_0 - self.dz

    @property
    def z_H(self):
        return self.z_0 + self.dz

    @property
    def z_s(self):
        if isinstance(self.z_L, (int, float, np.ndarray)) and isinstance(self.z_H, (int, float, np.ndarray)):   
            return torch.tensor([self.z_L, self.z_H]).to(self.device)
        elif isinstance(self.z_L, torch.Tensor) and isinstance(self.z_H, torch.Tensor): 
            return torch.stack([self.z_L, self.z_H])
        else:   
            raise ValueError(f"Invalid z_L and z_H types: {type(self.z_L)}, {type(self.z_H)}")


    def init_var(self, **kwargs):
        # exogenous density
        self.gw_np = np.ones(self.nx)
        # type discretization
        self.xmin = 1 / self.nx / 2
        self.xmax = 1 - 1 / self.nx / 2
        self.ymin = 1 / self.ny / 2
        self.ymax = 1 - 1 / self.ny / 2
        self.x_grid = np.linspace(self.xmin, self.xmax, self.nx)
        self.dx = (self.xmax - self.xmin) / (self.nx - 1)
        self.y_grid = np.linspace(self.ymin, self.ymax, self.ny)
        self.dy = (self.ymax - self.ymin) / (self.ny - 1)
    
    def init_path(self):
        param_names = ["beta", "kappa", "rho", "delta", "phi", "nu", "lam_L", "lam_H", "dz", "eta", "xi"]  # all estimable parameters
        param_names += ["xi_e"] if hasattr(self.par, 'xi_e') and True else []  # only add xi_e if it's in par_range and not dealing with legacy files, not adding it when it's only a constant
        param_names += ["c"] if hasattr(self.par_range, 'c') else []  # only add c if it's in par_range, not adding it when it's only a constant
        # param_names += ["xi_e"] if hasattr(self.par_range, 'xi_e') else []
        self.path = (
            f"nx{self.nx}ny{self.ny}{self.prod_type}" + format_params(self, param_names)
        )
                
        bx_str = format_params(self, ['b'])
        self.path = self.path + bx_str

        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def init_torch(self):
        """To avoid triggering CUDA initialization in multiprocessing"""
        nx, ny = self.nx, self.ny
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=(nx * ny), scramble=True)
        self.gw = torch.ones(self.nx).to(self.device)
        # The distribution of types is discretized to perform a midpoint Riemann sum with n subdivisions
        self.types_x = torch.linspace(1 / self.nx / 2, 1 - 1 / self.nx / 2, self.nx, device=self.device)
        self.types_y = torch.linspace(1 / self.ny / 2, 1 - 1 / self.ny / 2, self.ny, device=self.device)
        self.unif_x = torch.ones(self.types_x.shape[0])
        self.unif_y = torch.ones(self.types_y.shape[0])

    def calc_f(self, x, y):
        """Production function (numpy version)
        """
        if self.prod_type == 'CES':
            return 0.6 + 0.4 * (np.sqrt(x) + np.sqrt(y)) ** 2
        elif self.prod_type == 'EXP':
            return np.exp(x * y)
        elif self.prod_type == 'NEITHER':
            if (x <= 0.4):
                return 0.4 + y * (x + 0.6)
            else:
                return 0.4 + np.sqrt(np.square(x - 0.4) + np.square(y))
        elif self.prod_type == 'LR':
            # return (0.003 + 2.053 * x - 0.140 * y + 8.035 * x ** 2 - 1.907 * y ** 2 + 6.596 * x * y) * self.f0
            return (0.6 + .053 * x - 0.140 * y + .035 * x ** 2 - .5 * y ** 2 + 1.596 * x * y) * self.f0
        else:
            raise ValueError("prod_type must be 'CES', 'EXP', 'NAM', 'NEITHER', 'LR'.")

    def f_torch(self, x, y):
        """production function (torch)
        """
        if self.prod_type == 'CES':
            # return interpolator.interpolate(x, y)
            return 0.6 + 0.4 * (torch.sqrt(x) + torch.sqrt(y)) ** 2
        elif self.prod_type == 'EXP':
            return torch.exp(x * y)
        elif self.prod_type == 'NEITHER':
            if (x <= 0.4):
                return 0.4 + y * (x + 0.6)
            else:
                return 0.4 + torch.sqrt(torch.square(x - 0.4) + torch.square(y))
        elif self.prod_type == 'LR':
            # return (0.003 + 2.053 * x - 0.140 * y + 8.035 * x ** 2 - 1.907 * y ** 2 + 6.596 * x * y) * self.f0
            return (0.6 + .053 * x - 0.140 * y + .035 * x ** 2 - .5 * y ** 2 + 1.596 * x * y) * self.f0
        else:
            raise ValueError("prod_type must be 'CES', 'EXP', 'NAM', 'NEITHER', 'LR'.")

    def m(self, W, V):
        """Cobb-Douglas meeting function, nesting Lise&Robin(2017)'s Leontief one.
        """
        kappa, nu = self.kappa, self.nu
        return kappa * (W ** nu) * (V ** (1 - nu))

    def tensor_loader(self, name:str, sub_path='', to_np=False, device='cuda:0', dtype=None):
        """
        Load tensor from self.path+name, move to device, set to float32
        If the file ends with .npy, use np.load, otherwise use torch.load
        """
        dtype = torch.float32 if dtype is None else dtype
        full_path = os.path.join(self.path, sub_path + name)

        if name.endswith('.npy'):
            data = np.load(full_path)
            if not to_np:
                data = torch.from_numpy(data).to(device).to(dtype)
        elif name.endswith(('.pt', '.pth')):
            data = torch.load(full_path, map_location=device, weights_only= False).to(dtype)
            if to_np:
                data = data.detach().cpu().numpy()

        return data
