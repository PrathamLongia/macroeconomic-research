import itertools
import os
import pickle
import re
import time
from typing import Literal
import pandas as pd

import h5py
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from structures import Identity, Parameters, Ranges
from torch.utils.data import DataLoader, Dataset
from train_nn import Master_PINN_S, Train_NN
#from simulate import ergodics
from simulate_class import Simulation
import matplotlib.pyplot as plt



class Surrogate_NN(torch.nn.Module):
    """Maps parameter sets to ergodic aggregates"""
    def __init__(self, 
                 n_par=0,  # number of parameters, including fixed ones
                 n_out=0,  # number of output variables
                 moment_structure=None,  # dictionary of moment types and their indices in the output
                 activation=torch.nn.ReLU(),
                 nn_width=64,  # 72
                 nn_num_layers=4,
                 ):
        super(Surrogate_NN, self).__init__()
        self.moment_structure = moment_structure
        layers = [torch.nn.Linear(n_par, nn_width), activation]
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(activation)

        layers.append(torch.nn.Linear(nn_width, n_out))
        self.net = torch.nn.Sequential(*layers)
        for i in range(0, nn_num_layers):
            torch.nn.init.xavier_normal_(self.net[2 * i].weight)  # , gain=1.5)

    def forward(self, X, device=None):
        output = self.net(X)
        output = output.to(device) if device is not None else output
        if device == 'cpu':
            output = output.detach().numpy()
        if self.moment_structure is None:
            return output
        else:
            return {k: output[:, v] for k, v in self.moment_structure.items()}


class ChunkedDataset(Dataset):
    """
    Dataset that loads data in chunks from HDF5 files.
    Data is pre-generated and saved in multiple HDF5 files, but only loaded into memory when needed.
    """
    def __init__(self, data_dir: str, max_chunk_idx: int, split: Literal['train', 'val'], chunk_size: int = 1000, device="cuda:0"):
        """
        Args:
            data_dir (str): Directory containing the HDF5 data files
            chunk_size (int): Number of samples per chunk(file)
            max_chunk_idx (int): Maximum chunk index to load, preventing loading more chunks with different shape(N,T) were generated in previous runs, which will cause error
        """
        self.data_dir = os.path.join(data_dir, split)
        self.device, self.chunk_size, self.max_chunk_idx = device, chunk_size, max_chunk_idx

        # Find and sort all chunk files that match 'chunk_pattern' and are within the specified index range
        chunk_pattern = re.compile(r'chunk_(\d+)\.h5')
        self.chunk_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if chunk_pattern.match(f) and int(chunk_pattern.match(f).group(1)) <= self.max_chunk_idx
        ])

        # Total number of chunks
        self.total_chunks = len(self.chunk_files)
        
        # Get shape information from the first chunk file to determine data dimensions
        with h5py.File(os.path.join(self.data_dir, self.chunk_files[0]), 'r') as f:
            self.par_shape = f['parameters'].shape[1:]
            self.moments_shape = f['moments'].shape[1:]
        
        # Calculate the total size of the dataset, not including the last chunk initially
        self.total_size = (self.total_chunks - 1) * chunk_size 
        # Add the exact size of the last chunk, since it may not be full
        with h5py.File(os.path.join(self.data_dir, self.chunk_files[-1]), 'r') as f:
            self.total_size += len(f['parameters'])
            
        self.current_chunk_idx = -1  # Cache for current chunk
        self.current_chunk_data = None

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.total_size

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset by dynamically loading the necessary chunk."""

        # Determine which chunk the requested sample belongs to
        chunk_idx = idx // self.chunk_size
        # Index within the chunk
        local_idx = idx % self.chunk_size 
        
        # If the required chunk is not already loaded, load it into memory
        if chunk_idx != self.current_chunk_idx:
            chunk_file = os.path.join(self.data_dir, self.chunk_files[chunk_idx])
            with h5py.File(chunk_file, 'r') as f:
                self.current_chunk_data = {
                    'parameters': torch.from_numpy(f['parameters'][:]).float(),  # Load parameters as a PyTorch tensor
                    'moments': torch.from_numpy(f['moments'][:]).float()  # Load moments as a PyTorch tensor
                }
            self.current_chunk_idx = chunk_idx  # Update the cached chunk index
        
        # Return the requested sample from the loaded chunk
        return (
            self.current_chunk_data['parameters'][local_idx],
            self.current_chunk_data['moments'][local_idx]
        )


class Calibrator:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_moments = sum(len(moments) for moments in self.target_moments.values())  # total number of moments of all variables
        self.moment_structure = self.moment_struct()
        self.target_weight = self.default_target_weight() if self.target_weight is None else self.target_weight
        self.target_weight = self.normalize_target_weight(self.target_weight)  # normalize the target weight
        self.target_vars = set(itertools.chain.from_iterable(
                [self.target_moments[moment_type] for moment_type in self.target_moments]
            ))  # aggregate variables that have at least one moment target
        
        
    def moment_struct(self):
        """
        Create a dictionary that maps moment_type to a slice of target moments tensor.
        """
        structure = {}
        start = 0
        for moment_type in self.moment_types:
            if moment_type in self.target_moments:
                end = start + len(self.target_moments[moment_type])
                structure[moment_type] = slice(start, end)
                start = end
        return structure


    def default_target_weight(self):
        """
        Default target weight for each moment
        """
        return {
            key: {sub_key: 1.0 / len(self.target_moments[key]) for sub_key in self.target_moments[key]}
            for key in self.target_moments
        }


    def normalize_target_weight(self, target_weight):
        """Normalize the target weight to sum up to 1."""
        total_sum = sum(value for key in target_weight for value in target_weight[key].values())
        return {
            key: {sub_key: value / total_sum for sub_key, value in target_weight[key].items()}
            for key in target_weight
        }


    def set_instances(self, ct:Train_NN, pinn_S:Master_PINN_S, cfg:DictConfig, pinn_V_u=None, pinn_V_v=None):
        """
        Set model instances: Train_NN, Master_PINN_S, Master_PINN_V_u, Master_PINN_V_v.

        Set parameter instances.
        """
        self.ct = ct
        ## NN model instances
        self.pinn_S = pinn_S
        self.cfg = cfg
        self.pinn_V_u = pinn_V_u
        self.pinn_V_v = pinn_V_v


        ## Parameter instances
        self.par = ct.par
        self.n_par = len(ct.par)
        # parameters to be included in the grid
        self.relevant_pars = [key for key, value in self.ct.range.items() if not isinstance(value, Identity)]
        # indices of relevant parameters in self.ct.par
        self.relevant_par_idx = [self.ct.par.keys().index(key) for key in self.relevant_pars]
        self.n_relevant_par = len(self.relevant_pars)

        # Save path
        self.path = ct.path + '/calibrator'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    
    def steady_states(self, N=1000, printer=True):
        ct = self.ct
        N = np.min([N, ct.M_par_ss])
        start = time.time()
        par_draws, par_draw = ct.ss_pars(N)
        ct.saver.save_tensors({'ss_par_draw':par_draw})
        ct.steady_states_parallelizer(par_draws, tasks=[(par.z_0, par, i) for i, par in enumerate(par_draws)])
        print(f'SS par_draws takes {time.time() - start:.2f} seconds')
        del par_draws
        

    def moment_data(self, N=1000, T=5001, ct=None, pinn_S=None, erg_or_ss='ergodic', dt=0.01, pinn_V_u=None, pinn_V_v=None, N_erg=1, par_grid=None, par=None, load_mm=None, save=True, printer=True):
        """
        Generate aggregate moments of models with different parameter sets, as targets for getting calibrator mapping.

        If par_grid is provided, it should be a tensor of shape (n_grid_points^n_relevant_par, n_relevant_par), 
        and parameters missing from it will be set to the default values of self.ct.par.

        If par is provided, it should be a Parameters object of shape (N, n_par), and will be used as the parameter sets for generating moments.
        """
        ct = self.ct if ct is None else ct
        pinn_S = self.pinn_S if pinn_S is None else pinn_S
        pinn_V_u = self.pinn_V_u if pinn_V_u is None else pinn_V_u
        pinn_V_v = self.pinn_V_v if pinn_V_v is None else pinn_V_v

        device = self.device
        load_mm = self.load_moment_data if load_mm is None else load_mm
        N_erg = ct.N_erg if ct.N_erg is not None else N_erg
        if load_mm:
            print("Loading previously generated moment data...") if printer else None
            par_draw, filtered_moments, sim_moments = self.load_mm_data()
        else:
            print("Generating moment data...") if printer else None
            if erg_or_ss == 'ergodic':
                if par_grid is not None:
                    par_grid = par_grid.view(-1, self.n_relevant_par)  # (n_grid_points^n_relevant_par, n_relevant_par)
                    N = par_grid.shape[0]
                    par_draw = self.par_grid_to_par_draw(par_grid)  # supplement with default values of self.ct.par for missing parameters
                elif par is not None:
                    par.tensor()
                    par_draw = par
                else:
                    par_draw = ct.draw_parameters((N, 1), device) 
                par_draw = ct.c_target_tightness(par_draw) if ct.c_tightness else par_draw  # adjust c_target_tightness if chosen
                # Create simulation class
                sim = Simulation(ct, pinn_S, par_draw, cfg=self.cfg)
                sim_moments = sim.ergodics(dt, T, N_erg, if_plot = False, printer=printer)  # dict containing values of (N, )
                #sim_moments = ergodics(ct, pinn_S, par_draw, pinn_V_u, pinn_V_v, dt, T, N_erg, False, self.cfg, printer=printer)  # dict containing values of (N, )

            elif erg_or_ss =='ss':
                sim_moments = self.steady_states(ct, pinn_S, N, printer=printer)  # dict containing values of (N, )
            filtered_moments = list(itertools.chain.from_iterable(
                [sim_moments[moment_type][var] for var in self.target_moments[moment_type] if var in sim_moments[moment_type]]
                for moment_type, moment_slice in self.moment_structure.items()
            ))
            filtered_moments = torch.stack(filtered_moments, dim=1)  # (N, n_moments)
            if save:
                print("Saving moment data...")
                with open(self.path + '/filtered_moments.pkl', 'wb') as f:
                        pickle.dump((filtered_moments, sim_moments), f)
                ct.saver.save_tensors({'calibrator/par_draw_mm':par_draw})
        return par_draw, filtered_moments, sim_moments


    def load_mm_data(self):
        """Load parameters and filtered moments used for training mapping."""
        with open(self.path + '/filtered_moments.pkl', 'rb') as f:
            filtered_moments, sim_moments = pickle.load(f)
        par_draw = self.ct.tensor_loader("calibrator/par_draw_mm.pt") # Parameters object of (N, n_par), dict with n_moments values of (N,)
        return par_draw, filtered_moments, sim_moments
    

    def generate_and_save_chunks(self):
        """
        Generate simulation data and save it in chunks to HDF5 files.
        """
        print(f"Generating {self.N_sim} parameter draws in {self.N_sim//self.chunk_size} chunks, each with {self.chunk_size} samples...")
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        val_size = int(self.N_sim * self.val_ratio) if self.val_ratio is not None else 0  # number of validation data points
        train_size = self.N_sim  # val_size data is additional, no need to split, i.e. -val_size

        start = time.time()
        with open(os.path.join(self.path, 'chunk_data_log.txt'), 'w') as fo: pass  # clear train dataset log file of previous runs
        with open(os.path.join(self.path, 'val_chunk_data_log.txt'), 'w') as fo: pass  # clear val dataset log file of previous runs
        for chunk_start in range(0, train_size, self.chunk_size):
            chunk_size_actual = min(self.chunk_size, train_size - chunk_start)  # different from self.chunk_size iff is the last chunk
            chunk_id = chunk_start // self.chunk_size            
            par_draw, moments, _ = self.moment_data(chunk_size_actual, self.T_sim, load_mm=False, printer=False)
            chunk_file = os.path.join(train_dir, f'chunk_{chunk_id:05d}.h5')
            with h5py.File(chunk_file, 'w') as f:
                f.create_dataset('parameters', data=par_draw.cat().cpu().numpy())
                f.create_dataset('moments', data=moments.cpu().numpy())
            time_taken = time.time() - start 
            string = f"Saved chunk {chunk_id}: {chunk_start + chunk_size_actual}/{train_size} samples; time taken: {time_taken:.2f}s"
            print(string)
            fo = open(os.path.join(self.path, 'chunk_data_log.txt'), 'a')
            fo.write(string + '\n')
            fo.close()
        # Generate and save validation data 
        for chunk_start in range(0, val_size, self.chunk_size):  # Automatically jumped when self.val_ratio is None, i.e. val_size=0.
            chunk_size_actual = min(self.chunk_size, val_size - chunk_start)
            chunk_id = chunk_start // self.chunk_size
            par_draw, moments, _ = self.moment_data(chunk_size_actual, self.T_sim, load_mm=False, printer=False)
            chunk_file = os.path.join(val_dir, f'chunk_{chunk_id:05d}.h5')
            with h5py.File(chunk_file, 'w') as f:
                f.create_dataset('parameters', data=par_draw.cat().cpu().numpy())
                f.create_dataset('moments', data=moments.cpu().numpy())
            time_taken = time.time() - start 
            string = f"Saved validation chunk {chunk_id}: {chunk_start + chunk_size_actual}/{val_size} samples; time taken: {time_taken:.2f}s"
            print(string)
            fo = open(os.path.join(self.path, 'val_chunk_data_log.txt'), 'a')
            fo.write(string + '\n')
            fo.close()


    def get_chunked_dataloader(self, batch_size, split='train', shuffle=True, num_workers=10, device="cuda"):
        """
        Create a DataLoader for the chunked dataset.
        """
        max_chunk_index = int(np.ceil(self.N_sim / self.chunk_size)) - 1 if split == 'train' else int(np.ceil(self.N_sim * self.val_ratio / self.chunk_size)) - 1
        dataset = ChunkedDataset(self.data_dir, max_chunk_index, split, self.chunk_size, self.device)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers if torch.cuda.device_count() > 0 else 0,
            pin_memory=True if torch.cuda.device_count() > 0 else False,
            generator=torch.Generator(device=device)
        )


    def mapping(self, epochs=10000, batch_size=256, loss_threshold=1e-7, save_freq=10):
        """
        Train a NN to learn a mapping from parameter space to ergodic aggregate moments.
        """
        print("Getting a mapping from parameter space to ergodic aggregate moments")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_dir = os.path.join(self.path, 'data_chunks')
        if not self.load_moment_data:
            self.generate_and_save_chunks()

        # Create dataloaders for training (and validation)
        dataloader = self.get_chunked_dataloader(batch_size=batch_size, device=device)
        if self.val_ratio > 0:
            val_dataloader = self.get_chunked_dataloader(batch_size=min(batch_size, int(self.val_ratio*self.N_sim)), split='val', device=device)
            print(f"Training calibrator NN w/ {self.N_sim} training samples and {self.N_sim*self.val_ratio} validation samples using chunked data loading...")
        else:
            print(f"Training calibrator NN w/ {self.N_sim} samples using chunked data loading...")

        # Initialize the NN model
        model = Surrogate_NN(self.n_par, self.n_moments, self.moment_structure).to(device)

        # Set optimizer, lr scheduler, and loss function
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1)
        criterion = torch.nn.MSELoss()

        # Write loss values to log file
        with open(os.path.join(self.path, 'calibrator_loss.txt'), 'w') as fo: pass 
        
        start_time = time.time()

        # Main training loop
        for epoch in range(epochs + 1):
            for i_batch, (batch_par, batch_moments) in enumerate(dataloader):
                optimizer.zero_grad() 
                # Forward pass: Predict moments from parameters
                output = model(batch_par.to(device))

                # Compute loss as the sum of MSE losses for all moments
                loss = sum(criterion(output[k], batch_moments.to(device)[:, v]) for k, v in self.moment_structure.items())

                # Backpropagation
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                # Early stopping if loss is below the threshold
                if loss < loss_threshold:
                    print(f"Epoch {epoch}, batch {i_batch}, Loss: {loss.item()}, at {time.time() - start_time:.2f}s\nCalibrator NN Converged.")
                    torch.save({
                        'calibrator_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(self.path, 'calibrator_final.pt'))
                    print("Calibrator NN training complete, converged.")
                    return model

                # Log loss every 100 batches
                if i_batch % 100 == 0:
                    log_str = f"Epoch {epoch}, batch {i_batch}, Loss: {loss.item()}, at {time.time() - start_time:.2f}s"
                    with open(os.path.join(self.path, 'calibrator_loss.txt'), 'a') as fo:
                        fo.write(log_str + '\n')
                    print(log_str)

            # Save model checkpoint every `save_freq` epochs
            if save_freq is not None and epoch % save_freq == 0:
                savepath = os.path.join(self.path, f"calibrator_epoch_{epoch}.pt")
                torch.save({
                    'calibrator_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)

            # Validate every 5 epochs if validation data is available
            if self.val_ratio > 0 and epoch % 5 == 0 and epoch != 0:
                val_loss = 0
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    for val_batch_par, val_batch_moments in val_dataloader:
                        val_output = model(val_batch_par.to(device))
                        val_loss += sum(criterion(val_output[k], val_batch_moments.to(device)[:, v]) for k, v in self.moment_structure.items())

                val_loss /= len(val_dataloader)  # Compute average validation loss
                print(f"Epoch {epoch}, batch {i_batch}/{len(dataloader)}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
                model.train()  # Set model back to training mode

        torch.save({
            'calibrator_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.path, 'calibrator_final.pt'))

        print("Calibrator NN training complete, did not converge.")
        return model
    

    def load_mapping(self):
        """
        Load existing calibrator NN model.
        """
        print("Loading calibrator mapping--NN...")
        model = Surrogate_NN(self.n_par, self.n_moments, self.moment_structure).to(self.device)
        checkpoint = torch.load(os.path.join(self.path, "calibrator_final.pt"), map_location=self.device)
        model.load_state_dict(checkpoint['calibrator_state_dict'])
        model.eval()
        return model


    def par_grid_to_par_draw(self, par_grid:torch.Tensor, to_Parameters=True):
        """
        Convert a tensor of shape (n_grid_points^n_relevant_par, n_relevant_par) to a Parameters|dict,
        with values in the corresponding parameters, and with those of self.ct.par in the rest parameters.
        Each value is a tensor of shape (n_grid_points^n_relevant_par, 1)
        """
        new_params_dict = {}
        for param in self.ct.par.keys():
            if param in self.relevant_pars:
                idx = self.relevant_pars.index(param)
                new_params_dict[param] = par_grid[:, idx].unsqueeze(1)
            else:
                default_value = self.ct.par.__dict__[param]
                if par_grid.size(0) > 1:
                    new_params_dict[param] = torch.full((par_grid.size(0),1), default_value)
                else:
                    new_params_dict[param] = default_value
        if to_Parameters:
            return Parameters(new_params_dict)
        else:
            return new_params_dict  # values are a total of n_par tensors of shape (n_grid_points^n_relevant_par, 1)

    def par_grid(self, n_grid_point:int=None, bounds:dict=None):
        """
        Generate a parameter grid of shape (n_grid_point, n_grid_point,..., n_grid_point, n_relevant_par), 
        n_grid_point^n_relevant_par cooridnates each parameter to be calibrated
        """
        n_grid_point = self.n_grid_point if n_grid_point is None else n_grid_point
        bounds = self.ct.range.limits() if bounds is None else bounds
        bounds = {key: bounds[key] for key in self.relevant_pars}  # only input estimated parameters to calibrator, drop the rest parameters
        param_ranges = [torch.linspace(bounds[param].lower_bound, bounds[param].upper_bound, n_grid_point).to(self.device) for param in bounds]
        param_grids = torch.meshgrid(*param_ranges, indexing='ij')
        return torch.stack(param_grids, dim=-1)  # (n_grid_point, n_grid_point,..., n_grid_point, n_relevant_par), n_grid_point^n_relevant_par point cooridnates each parameter        


    def moment_distance(self, par=None, model=None, sim_moment=None, return_prediction=False, sum_dim=None):
        """
        Distance between simulated|(model moments on par) and target_moments for a given parameter set.
        One and only one of (par,model) or sim_moment must be provided.
        In the former case, par is a list of n_par parameter values, or a tensor of shape (N, n_par).
        In the latter case, sim_moment is a tensor of shape (N, n_moments).
        """
        device = self.device

        if model is not None:
            par = torch.tensor(par, dtype=torch.float32).reshape(-1, self.n_par).to(device) if isinstance(par, list|np.ndarray) else par.to(device)
            predicted_moments = model(par)
        elif sim_moment is not None:
            predicted_moments = sim_moment  # if simluated_moment is provided, mapping is not used
        else:
            raise ValueError("One and only one of model or mm_sim must be provided.")
        
        targets = torch.tensor([self.target_moments[moment_type][var] for moment_type in self.target_moments for var in self.target_moments[moment_type]], 
                            dtype=torch.float32, device=device)
        weights = torch.tensor([self.target_weight[moment_type][var] for moment_type in self.target_moments for var in self.target_moments[moment_type]], 
                            dtype=torch.float32, device=device)
        predictions = torch.cat([predicted_moments[moment_type] for moment_type in self.target_moments], dim=1) if sim_moment is None else sim_moment.squeeze()
        dis = torch.sum(((predictions - targets) / targets * weights) ** 2, dim=sum_dim) if self.if_relative_dis else torch.sum(((predictions - targets) * weights) ** 2, dim=sum_dim)
        
        if return_prediction:
            predictions = {moment_type: {var: predicted_moments[moment_type][:, i] for i, var in enumerate(vars) } for moment_type, vars in self.target_moments.items()}
            predictions = {moment_type: {var: (
                        float(predictions[moment_type][var]) if isinstance(predictions[moment_type][var], np.ndarray) and predictions[moment_type][var].shape == (1,) 
                        else predictions[moment_type][var].item() if isinstance(predictions[moment_type][var], torch.Tensor) and predictions[moment_type][var].shape == (1,) 
                        else predictions[moment_type][var]
                    ) for var in vars} for moment_type, vars in self.target_moments.items()}
            return dis, predictions
        else:
            return dis  # scalar|(N,)|(,n_moments) depending on sum_dim
    
    def find_optimal_par(self, model):
        """
        Finds the optimal parameter values by minimizing the distance between simulated moments
        (computed using the given model) and target moments.

        Returns a dictionary containing the optimized parameter values.
        """
        print("Finding optimal parameters using mapping...")
        start = time.time()
        
        # Set initial guess to be parameter values
        initial_guess = dict(self.ct.par.items())

        self.ct.range = Ranges(self.par, self.ct.par_range, device='cpu')  # extrapolate on search range enlarged compared that used for training(solving the model)
        bounds = self.ct.range.limits()

        # Initialize parameters as trainable Pytorch tensors
        parameters = torch.nn.ParameterDict()
        for name, value in initial_guess.items():  # inline initialization will change the order of parameters
            parameters[name] = torch.nn.Parameter(torch.tensor([value], device=self.device, dtype=torch.float32))
        
        # Optimization loop to find the best parameters
        optimizer = optim.AdamW(parameters.values(), lr=0.001)
        for _ in range(10000):
            optimizer.zero_grad()
            par = torch.cat([param for param in parameters.values()]).unsqueeze(0)
            loss = self.moment_distance(par, model)
            loss.backward()
            optimizer.step()
            self.constrain_parameters(parameters.items(), bounds)
        optimal_par = {name: param.item() for name, param in parameters.items()}
        
        print(f'Time used for finding optimal parameters: {time.time() - start:.2f} seconds')
        return optimal_par

    def export_calibration_results(self):
        '''
        Export optimal moment results to excel file
        '''
        rows = []
        for moment_type, vars in self.optimal_complete_mm.items():
            for var_name, optimal_value in vars.items():
                optimal_value = optimal_value.mean().item() if hasattr(optimal_value, 'mean') else optimal_value.item() if isinstance(optimal_value, np.ndarray) else optimal_value
                target_value = self.target_moments.get(moment_type, {}).get(var_name, '')
                target_weight = self.target_weight.get(moment_type, {}).get(var_name, 0)

                pct_diff = ''
                if target_value != '' and target_weight != 0:
                    pct_diff = f"{((optimal_value - target_value) / target_value) * 100:.2f}%"

                moment_name = {'mean': f"E[{var_name}]", 'std': f"σ[{var_name}]", 'autocorr': f"ρ[{var_name}]"}.get(moment_type, f"{moment_type}[{var_name}]")
                rows.append([moment_name, target_value, optimal_value, pct_diff])

        df = pd.DataFrame(rows, columns=['Moment', 'Data', 'Optimal Simulation', 'Relative Diff%(if targeted)'])
        output_path = os.path.join(self.path, f'calibration_output.xlsx')
        df.to_excel(output_path, index=False)
        print(f"Moments comparison exported to {output_path}")

    def constrain_parameters(self, parameters, bounds):
        """
        Constrains parameter values to remain within specified bounds.
        Modifies parameters in-place.
        """
        with torch.no_grad():
            for name, param in parameters:
                if name in bounds:
                    lower = bounds[name].lower_bound.to(self.device)
                    upper = bounds[name].upper_bound.to(self.device)
                    param.data.clamp_(lower, upper)

    def load_calibration_result(self):
        """
        Loads previously saved calibration results from a .pkl file.
        """
        with open(os.path.join(self.path, 'calibration_result.pkl'), 'rb') as f:
            # Load optimal parameters, distance, surrogate predicted moments, and simulated moments
            self.optimal_par, self.optimal_dis, self.optimal_moments, self.optimal_complete_mm = pickle.load(f)

    def print_opt(self, saver=True):
        """
        Print and save .txt file of optimal parameters and moments they produce
        """
        string = f'Using parameters:\n{dict(self.optimal_par.items())}\n'
        string += f' with simulated moments:\n{self.optimal_complete_mm}\n and surrogate predicted moments\n{self.optimal_moments}\nyielding'
        string += ' relative ' if self.if_relative_dis else ' absolute '
        string += f' surrogate predicted distance: {self.optimal_dis}'
        print(string)
        if saver:
            string += f'\n\nct.par_range:\n{self.ct.par_range}\n'
            string += f'\nct.par:\n{self.ct.par}\n'
            fo = open(os.path.join(self.path, 'opt_calibrate_result.txt'), "w")
            fo.write(string)

    def calibrate(self):
        """
        Runs the calibration process and returns the optimal parameters.
        """
        # Load saved mapping if available
        if self.load_saved_mapping: 
            model = self.load_mapping()
        # Otherwise create a new mapping NN model
        else: 
            model = self.mapping(self.epochs, self.batch_size, self.loss_threshold, self.save_freq)

        # Load previous calibration results if available.
        if self.load_ca_result:
            self.load_calibration_result()
        # Otherwise, perform calibration
        else:
            # Find optimal parameters using the model
            self.optimal_par = self.find_optimal_par(model)

            # Compute moment distance and get predicted moments
            self.optimal_dis, self.optimal_moments = self.moment_distance(
                                    list(self.optimal_par.values()), model, return_prediction=True)
            
            _, _, self.optimal_complete_mm = self.moment_data(par=Parameters(self.optimal_par), load_mm=False, save= False) 
            self.optimal_complete_mm = {mm_type: {var: tensor.item() for var, tensor in vars.items()} for mm_type, vars in self.optimal_complete_mm.items()}

            # Save calibration results
            with open(os.path.join(self.path, 'calibration_result.pkl'), 'wb') as f:
                pickle.dump((self.optimal_par, self.optimal_dis, self.optimal_moments, self.optimal_complete_mm), f)
        
        # Print and save the calibration results to a text/excel file
        self.export_calibration_results()
        self.print_opt()

     