# With some class modified from Kase, Melosi & Rottner (2022): "Estimating nonlinear heterogeneous
# agents models with neural networks". CEPR Discussion Paper.

import torch
from torch.distributions import constraints
from typing import Dict
from omegaconf import DictConfig
import numpy as np
from copy import deepcopy
# %% Base class for the parameters of the model
class Element(object): 
    def __init__(self, dict: Dict, callback=None) -> None:
        self.callback = callback  # callback function for updating parameters in Environemt class when parameters are updated in Parameters class
        for key, value in dict.items():
            # if isinstance(value, float):
            #     value = torch.tensor([value])
            setattr(self, key, value)  
    
    def exclude_attrs(self):
        return ['callback']    
    
    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key not in self.exclude_attrs() and self.callback is not None:
            self.callback(key, value)

    def __setitem__(self, key, value):
        setattr(self, key, value)  # already callbacked to Environment in __setattr__

    def __getitem__(self, idx):
    # def __getitem__(self, key):
        # return getattr(self, key)
        new_dict = {}
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                new_dict[key] = value[idx]
            else:
                new_dict[key] = value
        return Element(new_dict)
    
    def __len__(self, count_exclude_attrs=False):
        if count_exclude_attrs:
            return len(self.__dict__)
        else:
            return len(self.__dict__) - len(self.exclude_attrs())

    def __contains__(self, key):
        return key in self.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        output = ""
        for key, value in self.__dict__.items():
            output += f"{key}: {value}\n"
        return output

    def size(self):
        items = self.items()
        items = [item for item in items if item[0] not in self.exclude_attrs()]
        first_attr_type = type(items[0][1])
        if first_attr_type == torch.Tensor:
            N = items[0][1].size(0)  # assume all tensors have the same size
            length = len(self)
            return torch.Size([N, length])
        elif first_attr_type in (float, int):
            return torch.Size([1, len(self)])
        else:
            return f'Unsupported type of parameter, must be float or tensor.'
    
    @property
    def shape(self):
        return self.size()

    def numpy(self):
        new_dict = {}
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                if isinstance(value, torch.Tensor):
                    new_dict[key] = value.detach().cpu().numpy()
                else:
                    new_dict[key] = value
        return Element(new_dict)

    def tensor(self, device='cuda:0', dtype=torch.float32):
        """set all float attributes to tensor attributes"""
        for key, value in self.__dict__.items():
            if isinstance(value, (float, int)):
                value = torch.tensor([value], dtype=dtype, device=device)
            setattr(self, key, value)

    def update(self, dict: Dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return [key for key, value in self.__dict__.items() if key not in self.exclude_attrs()]

    def values(self):
        return [value for key, value in self.__dict__.items() if key not in self.exclude_attrs()]

    def list(self):
        return self.values()

    def items(self):
        return [(key,value) for key, value in self.__dict__.items() if key not in self.exclude_attrs()]
    
    def cat(self, device='cuda:0', dtype=torch.float32):
        """move parameters to device, concatenate them by dim=-1, return tensor"""
        values = [value for key, value in self.__dict__.items() if key not in self.exclude_attrs()]
        if isinstance(values[0], float):
            return torch.cat([torch.tensor([value], device=device, dtype=dtype) for value in values], dim=-1)
        elif isinstance(values[0], torch.Tensor):
            values = [value.view(-1, 1) if value.dim() == 1 and value.shape != (1,) else value for value in values]
            return torch.cat([value for value in values], dim=-1).to(device, dtype=dtype)
        else:
            raise ValueError("Unsupported type of parameter, must be float or tensor.")

    def to(self, device):
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                if isinstance(value, (torch.distributions.Distribution,)):
                    for param_key, param_value in value.__dict__.items():
                        if isinstance(param_value, torch.Tensor):
                            setattr(value, param_key, param_value.to(device))
                else:
                    setattr(self, key, value.to(device))
        return self


    def device(self):
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                return value.device
        return None

    def requires_grad(self, requires_grad=True):
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                value.requires_grad = requires_grad
        return self

    def zero_grad(self):
        for key, value in self.__dict__.items():
            if key not in self.exclude_attrs():
                value.grad = None
        return self


# %% Class for the parameters
class Parameters(Element):
    def expand(self, shape):
        return Parameters(deepcopy({key: value.expand(shape) if isinstance(value, torch.Tensor) else torch.tensor(value).expand(shape) if key not in self.exclude_attrs() else value 
             for key, value in self.__dict__.items()})
        )
    def repeat_interleave(self, repeats, dim=0):
        return Parameters(deepcopy({key: value.repeat_interleave(repeats, dim) if isinstance(value, torch.Tensor) else torch.tensor(value).repeat_interleave(repeats, dim) if key not in self.exclude_attrs() else value 
             for key, value in self.__dict__.items()})
        )


# %% Identity class for fixed parameters
class Identity(object):
    def __init__(self, value, device='cuda:0', dtype=torch.float32) -> None:
        self.value = torch.tensor(value, device=device, dtype=dtype)

    def __str__(self):
        return f"Identity({str(self.value.item())})"

    def sample(self, shape):
        identity_sample = self.value.expand(shape)
        return identity_sample

    @property
    def support(self):  
        return constraints.interval(self.value-1e-15, self.value+1e-15)  # usually indiscrenible for double precision

    @property
    def low(self):  
        return self.support.lower_bound

    @property
    def high(self):  
        return self.support.upper_bound

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


# %% Class for the priors
class Ranges(Element):
    def __init__(self, par_dict: Dict|DictConfig|Parameters, priors_dict: Dict | DictConfig, callback=None, device='cuda:0') -> None:
        self.callback = callback  # callback function for updating parameters in Environemt class when parameters are updated in Ranges class
        self.priors_dict = priors_dict
        self.device = device
        for key, value in par_dict.items():
            setattr(self, key, value)
    
    def exclude_attrs(self):
        return super().exclude_attrs() + ['priors_dict', 'device']
    
    def __len__(self, count_exclude_attrs=False):
        return super().__len__(count_exclude_attrs)
    
    def __setattr__(self, key, value):
        if key not in self.exclude_attrs():
            if hasattr(self, 'priors_dict'):
                if key in self.priors_dict:
                    if isinstance(value, torch.distributions.Distribution):
                        pass  # update to a new distribution object
                    else:  # isinstance(value, float|torch.Tensor|np.ndarray|int)
                        value = self.priors_dict[key]  # must be a distribution object for those in priors_dict
                else:
                    if isinstance(value, float|torch.Tensor|np.ndarray|int):
                        value = Identity(value, device=self.device)
                    elif isinstance(value, Identity):
                        pass  # update to a new Identity object
                    else:
                        raise ValueError(f"Unsupported type of parameter {key}, must be float|torch.Tensor|np.ndarray|int|Identity.")
        object.__setattr__(self, key, value)  # no need to callback to Environment. Weonly change par, and callback in Environment._update_attribute will automatically update Identity values

    def cat(self):
        pass

    def limits(self, device="cuda:0"):
        return {key: value.support.to(device) if isinstance(value.support, torch.Tensor) else value.support 
                for key, value in self.__dict__.items() if key not in self.exclude_attrs()}

    def low_tensor(self):  # only used in normalize layer
        return torch.tensor([value.low for value in self.__dict__.values() if value not in self.exclude_attrs()])

    def high_tensor(self):  # only used in normalize layer
        return torch.tensor([value.high for value in self.__dict__.values() if value not in self.exclude_attrs()])
    
    def sample(self, shape, device="cuda:0"):
        par_draw = {key: value.sample(shape).to(device) for key, value in self.__dict__.items() if key not in self.exclude_attrs()}
        return Parameters(par_draw)
