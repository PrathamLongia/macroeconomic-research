# -*- coding: utf-8 -*-
# @author: Runhuan Wang

import torch
import os
import numpy as np


# %% Function to format number
def format_number(value, max_decimal_places=6):
    """format number to string to trail unnecessary 0 and . in the end, and with max_decimal_places decimal places if it is float, otherwise return the string of the number."""
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        else:
            formatted = f"{value:.{max_decimal_places}f}"
            formatted = formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
            return formatted
    else:
        return format_number(float(value))  # not float
# %% Function to format path
def format_params(ct, param_names):
    """Format estimable parameter by ranges or values."""
    formatted_params = []
    for param_name in param_names:
        if hasattr(ct.par_range, param_name):
            param_range = getattr(ct.par_range, param_name)
            formatted_params.append(f"{param_name}{format_number(param_range.low)}-{format_number(param_range.high)}" if param_range else f"{param_name}{format_number(getattr(ct.par, param_name))}")
        else:
            if hasattr(ct.par, param_name):
                formatted_params.append(f"{param_name}{format_number(getattr(ct.par, param_name))}")
            else:
                raise ValueError(f"Parameter {param_name} not found in par or par_range or par.")
    return ''.join(formatted_params)
# %% Saver class to save tensors and numpy ndarray
class Saver:
    def __init__(self, path):
        self.path = path

    def save_tensors(self, tensors, par=None, z=None, suffix=''):
        """tensors:dict|Tensor. par:Parameter. z:absolute value, not index."""
        suffix = str(suffix)
        if z is not None and par is not None:
            if z == par.z_0:
                suffix += "_ss"
            elif z == par.z_0 + par.dz:
                suffix += "_high_z"
            elif z == par.z_0 - par.dz:
                suffix += "_low_z"
            else:
                raise ValueError(f"Unexpected value for z: {z}")
        if isinstance(tensors, dict):
            for name, tensor in tensors.items():
                filename = name + suffix + ".pt"
                torch.save(tensor, os.path.join(self.path, filename))
        elif isinstance(tensors, torch.Tensor):
            filename = suffix + ".pt"
            torch.save(tensors, os.path.join(self.path, filename))
        elif isinstance(tensors, list):
            local_vars = locals()
            for tensor in tensors:
                name = [var_name for var_name, var_val in local_vars.items() if var_val is tensor][0]
                filename = name + suffix + ".pt"
                torch.save(tensor, os.path.join(self.path, filename))

    def save_numpy(self, arrays, par=None, z=None, suffix=''):
        """arrays:dict|np.ndarray. par:Parameter. z:absolute value, not index."""
        suffix = str(suffix)
        if z is not None and par is not None:
            if z == par.z_0:
                suffix += "_ss"
            elif z == par.z_0 + par.dz:
                suffix += "_high_z"
            elif z == par.z_0 - par.dz:
                suffix += "_low_z"
            else:
                raise ValueError(f"Unexpected value for z: {z}")
        if isinstance(arrays, dict):
            for name, array in arrays.items():
                array = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array
                filename = name + suffix + ".npy"
                np.save(os.path.join(self.path, filename), array)
        elif isinstance(arrays, np.ndarray):
            filename = suffix + ".npy"
            np.save(os.path.join(self.path, filename), arrays)
        elif isinstance(arrays, list):
            local_vars = locals()
            for array in arrays:
                name = [var_name for var_name, var_val in local_vars.items() if var_val is array][0]
                filename = name + suffix + ".npy"
                np.save(os.path.join(self.path, filename), array)



# %% Class to analyze time series
class TimeSeries:
    def __init__(self):
        pass

    def mean_std_autocorr(self, tensors, burn_in=0, saver:Saver=None, names:list=None, suffixes:list=None, N_par=1):
        """
        Calculate the mean, standard deviation, and autocorrelation of a list of aggregates time series.
        :param
          tensors:  a list of tensors of shape (N, T_original), where N is the number of series and T_original is the length of the series.
          burn_in:  the number of initial periods to be discarded.
          saver:    a Saver object to save the sim_moments.
          names:    a list of names for the series.
          suffixes: a list of 3 suffixes for [means, stds, autocorrs].
        :return: a dictionary containing ALL the mean, standard deviation, and autocorrelation of (M, N, T), where T=T_original-burn_in.
        """
        concatenated_tensor = torch.stack(tensors, dim=0)  #  (M, N, T_original)
        del tensors
        torch.cuda.empty_cache()
        concatenated_tensor = concatenated_tensor[:, :, burn_in:]  #  (M, N, T)
        M, N, T = concatenated_tensor.shape
        suffixes = ['_mean', '_std', '_autocorr'] if suffixes is None else suffixes
        
        N = N // N_par  # N_erg, num of simulation each par draw
        concatenated_tensor = concatenated_tensor.view(M, N_par, N, T)  #  (M,N_par*N_erg,T)->(M,N_par,N,T)
        means = torch.mean(concatenated_tensor, dim=(2, 3))  #  (M,N_par)
        stds = torch.std(concatenated_tensor, dim=(2, 3))  #  (M,N_par)
        autocorrs = torch.sum((concatenated_tensor[:, :, :, 1:] - means[:, :, None, None]) * (concatenated_tensor[:, :, :, :-1] - means[:, :, None, None]), dim=(2, 3)) / ((N * (T - 1)) * stds**2)  #  (M,N_par)
        if N_par == 1 and saver is not None and names is not None:
                saver.save_numpy(dict(zip(names, means.tolist())), suffix=suffixes[0])
                saver.save_numpy(dict(zip(names, stds.tolist())), suffix=suffixes[1])
                saver.save_numpy(dict(zip(names, autocorrs.tolist())), suffix=suffixes[2])
       
        if names is not None:
            means = dict(zip(names, torch.unbind(means, dim=0)))
            stds = dict(zip(names, torch.unbind(stds, dim=0)))
            autocorrs = dict(zip(names, torch.unbind(autocorrs, dim=0)))
        sim_moments = {
            'mean': means,
            'std': stds,
            'autocorr': autocorrs
        }

        return sim_moments
