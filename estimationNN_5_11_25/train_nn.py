import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import time
from env import Environment
from torch.utils.data import DataLoader, TensorDataset

from structures import Parameters
from ss_surrogate import Steady_State_Surrogate
from ss_solver import SteadyStateSolver


class Master_PINN_S(torch.nn.Module):
    """Surplus NN function approximation"""
    def __init__(self, activation=torch.nn.Tanh(),
                 nn_width=50,
                 nn_num_layers=4,
                 n_x=2,
                 n_y=2,
                 n_par=0,  # number of parameters, including fixed ones
                 ):
        super(Master_PINN_S, self).__init__()
        # Construct an array of affine and activation functions.
        # Hidden layers:
        #   These are the hidden layers 1,2,..., nn_num_layers
        #   layers = [affine1, activation1, affine2, activation2, ...]
        layers = [torch.nn.Linear(3 + n_x * n_y + n_par, nn_width), activation]
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(activation)

        layers.append(torch.nn.Linear(nn_width, 1))
        # Sequentially execute the affine and activations function;
        # This constructs the neural network approximation.
        self.net = torch.nn.Sequential(*layers)
        for i in range(0, nn_num_layers):
            # Apply the xavier normalization at the affine layers.
            torch.nn.init.xavier_normal_(self.net[2 * i].weight)

    def forward(self, X):
        return self.net(X)


class Train_NN(Environment):
    """Subclass for solving the system using deep learning."""
    def __init__(self, **kwargs):
        # Initialize the parent class with kwargs
        super().__init__(**kwargs)

        # Set working directory
        self.working_dir = os.getcwd()
        print('Path:\n', self.working_dir + '\\' + self.path)

        # Extract parameters 
        par = self.par

        # Set initial firm density
        # letting gf all 1 would imply U=V when z=z_0: V+P=gf=gp+gv,gw=U+E,mean(gp)=P=E=mean(ge)
        self.init_gf = 1 + self.U_target * (self.init_v_u_ratio - 1)

        # Solve for deterministic steady state (DSS) w/o free entry using fixed point solver
        self.ss_solver = SteadyStateSolver(**kwargs)
        _, _, _, self.g0, new_c, _, _, _, _, _ = self.ss_solver.create_g_no_free_entry(self.z_0, np.ones(self.ny) * self.init_gf, par=par)
        self.c = new_c if self.calibrate_init_c else self.c
        print('Calibrated DSS(z0) c=', self.c)

        # Calculate U and V at DSS w/o free-entry
        g_e = np.mean(self.g0, axis=1)
        g_u = self.gw_np - g_e
        g_p = np.mean(self.g0, axis=0)
        g_v = self.init_gf - g_p
        self.V0 = np.mean(g_v, axis=0)
        self.U0 = np.mean(g_u, axis=0)
        print(f"U={self.U0 * 100:.4f}%,V={self.V0 * 100:.4f}%, at SS w/o free-entry, z= {self.z_0}")

    def solve_steady_state(self, par=None, printer=False):
        """Solve for DS for desired number of g, with one or more parameter sets in parallel"""
        par = self.par if par is None else par
        if not self.load_g_ss:
            for z in [par.z_0, par.z_0 - par.dz, par.z_0 + par.dz]:
                print("Starting DSS Fixed Point Algorithm: kappa = " + str(par.kappa) + ", z = " + str(z))
                if self.no_fe: self.ss_solver.create_g_no_free_entry(z, np.ones(self.ny) * self.init_gf, par=par, printer=printer)
                else: self.ss_solver.create_g_free_entry(z, par, printer=printer)
        
            start = time.time()
            par_draws, par_draw = self.ss_pars(self.M_par_ss)
            self.saver.save_tensors({'ss_par_draw':par_draw})
            self.ss_solver.compute_all_steady_states(par_draws)
            print(f'SS par_draws takes {time.time() - start:.2f} seconds')
            del par_draws
        else:
            print('Loading previous Steady States results, for multiple parameter sets')

    def ss_pars(self, M=14):
        """Return a list of Parameters objects for DSS"""
        print(f"{M} parameter sets for DSS")
        par_draw = self.draw_parameters((M, 1))
        return [Parameters({key: value[i].item() for key, value in par_draw.items()}) for i in range(M)], par_draw  # DSS are computed on CPUs

    def ss_surrogate_model(self, epochs=None, batch_size=16, loss_threshold=1e-7, save_freq=1000):
        """Train a neural network surrogate model to learn a mapping from parameter space to 
        steady state vector (g(x,y,z), s(x,y,z)), avoiding the need for expensive fixed-point
        computations during the main training loop.
        """
        
        print("Getting a mapping from parameter space to steady state g(x,y,z),s(x,y,z)...")
        device = self.device
        
        # Set epochs if not provided
        epochs = self.epochs_ss_surrogate if epochs is None else epochs
        
        # Load precomputed steady state parameter draws and solutions
        self.par_draw_ss = self.tensor_loader("ss_par_draw.pt")
        start = time.time()
        print("Training steady state surrogate neural network...")
        
        # Get parameter draws and corresponding high/low state solutions
        par_draw, g_highs, g_lows, s_highs, s_lows = self.par_draw_ss, self.g_highs, self.g_lows, self.s_highs, self.s_lows  # (M,n_par), (M,nx,ny), (M,nx,ny)
        
        # Stack and flatten the high/low state solutions for neural network input
        g_xyz = torch.stack([g_highs, g_lows], dim=3).flatten(start_dim=1)  # (M,nx,ny,2)->(M,nx*ny*2)
        s_xyz = torch.stack([s_highs, s_lows], dim=3).flatten(start_dim=1)  # (M,nx,ny,2)->(M,nx*ny*2)
        
        # Handle tightness constraint case by including c parameters
        if self.c_tightness:
            c_par = self.c_pars.view(self.M_par_ss, 1)    # (M,1)
            gsc = torch.cat([g_xyz, s_xyz, c_par], dim=1)  # (M,nx*ny*2*2+1)
        else:
            gsc = torch.cat([g_xyz, s_xyz], dim=1)  # (M,nx*ny*2*2)
        
        # Create dataset and dataloaders
        dataset = TensorDataset(par_draw.cat(), gsc)
        
        # Split into training and validation sets if validation is enabled
        if self.ss_surrogate_val:
            train_size = int(0.95 * len(dataset)) 
            val_size = len(dataset) - train_size 
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator(device=device))
            dataloader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True, generator=torch.Generator(device=device))
            val_dataloader = DataLoader(val_dataset, batch_size=min(batch_size, val_size), shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
        
        # Initialize model, optimizer, and scheduler
        model = Steady_State_Surrogate(n_par=len(self.par), nx=self.nx, ny=self.ny, c_tightness=self.c_tightness).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_surrogate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1)
        criterion = torch.nn.MSELoss()
        
        # Initialize loss log file
        with open(os.path.join(self.path, 'ss_surrogate_loss.txt'), 'w') as fo: pass 
        
        # Training loop
        for epoch in range(epochs + 1):
            # Batch training
            for batch_par, batch_gs in dataloader:
                optimizer.zero_grad()
                output = model(batch_par)
                loss = criterion(output, batch_gs) / len(batch_gs)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

            # Print training progress periodically
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], SS Surrogate NN Loss: {loss.item():4e}')
            
            # Validation if enabled
            if self.ss_surrogate_val and (epoch + 1) % 500 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_par, val_gs in val_dataloader:
                        val_output = model(val_par)
                        val_loss += criterion(val_output, val_gs).item()
                val_loss /= len(val_dataloader)
                print(f'         Val Loss: {val_loss:.4e}')
                model.train()

            # Log loss to file periodically
            if epoch % 100 == 0 and epoch != 0:
                fo = open(os.path.join(self.path, 'ss_surrogate_loss.txt'), 'a')
                string = "Iter %d: Loss = %.4e" % (epoch, loss.item())
                fo.write((string + '\n'))
                fo.flush()
            
            # Save model checkpoint periodically
            if save_freq is not None and epoch % save_freq == 0:
                model_save_name = f"ss_surrogate_NN_epoch_{epoch}.pt"
                savepath = os.path.join(self.path, model_save_name)
                torch.save({
                    'ss_surrogate_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, savepath)
            
            # Early stopping if loss threshold reached
            if (loss < loss_threshold):
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                print("SS Surrogate NN Converged.")
                break
        
        # Save final model
        torch.save({
            'ss_surrogate_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(self.path, f'ss_surrogate_final_c{int(self.c_tightness)}.pt'))
        
        print("SS Surrogate NN training complete.")
        print(f'SS Surrogate training took {time.time() - start:.2f} seconds')
        return model

    def load_ss_surrogate_model(self):
        """Loads a pre-trained neural network surrogate model for steady-state solutions.
            Returns the Steady_State_Surrogate model.
        """
        # Store number of parameters in the parameter set
        self.n_par = len(self.par)
        
        # Initialize the neural network model with matching architecture
        print("Loading SS surrogate--NN...")
        model = Steady_State_Surrogate(
            n_par=len(self.par),      # Number of parameters
            nx=self.nx,               # Number of worker types
            ny=self.ny,               # Number of firm types
            c_tightness=self.c_tightness  # Whether to include tightness constraint
        ).to(self.device)             
        
        # Load the saved model weights from checkpoint file
        checkpoint_path = os.path.join(self.path, f"ss_surrogate_final_c{int(self.c_tightness)}.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['ss_surrogate_state_dict'])
        
        return model


    def c_target_tightness(self, par=None):
        """
        Compute and update the target tightness parameter 'c' using a pre-trained surrogate model 
        for steady state solutions.
        
        Args:
            par: Optional parameter object. Uses self.par_draw if None.
            
        Returns:
            par: Updated parameter object with new 'c' value (if c_tightness enabled)
        """
        # Use default parameters if none provided
        par = self.par_draw if par is None else par
        
        if self.c_tightness:
            # Load pre-trained surrogate model for steady-state solutions
            self.ss_surrogate = self.load_ss_surrogate_model()
            self.ss_surrogate.eval()
            
            # Predict steady-state density solution using current parameters
            # Note: cat() concatenates parameters into model input format
            gsc = self.ss_surrogate(par.cat())
            
            # Extract tightness parameter (last dimension of output)
            c_par = gsc[..., -1].view(-1, 1).detach()
            c_par = c_par.squeeze(-1) if par.cat().dim() == 1 else c_par
            
            # Update parameters onject with new tightness value
            par.update({'c': c_par})
            del gsc, c_par
        return par


    def init_g_sample_space(self):
        # Construct lists of expected filenames for high/low g and surplus values
        g_highs_files = [f"g{i}_high_z.npy" for i in range(self.M_par_ss)]
        g_lows_files = [f"g{i}_low_z.npy" for i in range(self.M_par_ss)]
        s_highs_files = [f"surplus{i}_high_z.npy" for i in range(self.M_par_ss)]
        s_lows_files = [f"surplus{i}_low_z.npy" for i in range(self.M_par_ss)]
        
        # Conditionally include c_par files if c_tightness is enabled
        c_par_files = [f"c_par{i}_ss.npy" for i in range(self.M_par_ss)] if self.c_tightness else []
        
        # Determine which consolidated files to check for based on c_tightness
        files_check = ['g_highs.pt', 'g_lows.pt', 's_highs.pt', 's_lows.pt', 'c_pars.pt'] if self.c_tightness else ['g_highs.pt', 'g_lows.pt', 's_highs.pt', 's_lows.pt']
        all_files = g_highs_files + g_lows_files + s_highs_files + s_lows_files + c_par_files
        
        # Case 1: All individual result files exist
        if all(os.path.exists(self.path + '/' + file) for file in all_files):
            # Load and stack g_high values
            g_highs = [self.tensor_loader(file) for file in g_highs_files]
            self.g_highs = torch.stack(g_highs)  # Shape: (M, nx, ny)
            
            # Load and stack g_low values
            g_lows = [self.tensor_loader(file) for file in g_lows_files]
            self.g_lows = torch.stack(g_lows)  # Shape: (M, nx, ny)
            
            # Load and stack surplus high values
            s_highs = [self.tensor_loader(file) for file in s_highs_files]
            self.s_highs = torch.stack(s_highs)  # Shape: (M, nx, ny)
            
            # Load and stack surplus low values
            s_lows = [self.tensor_loader(file) for file in s_lows_files]
            self.s_lows = torch.stack(s_lows)  # Shape: (M, nx, ny)
            
            # Conditionally handle c_par values if c_tightness is enabled
            if self.c_tightness:
                c_pars = [self.tensor_loader(file) for file in c_par_files]
                self.c_pars = torch.stack(c_pars)  # Shape: (M,)
                # Save all loaded tensors in consolidated files
                self.saver.save_tensors({
                    'g_highs': self.g_highs, 
                    'g_lows': self.g_lows, 
                    's_highs': self.s_highs, 
                    's_lows': self.s_lows, 
                    'c_pars': self.c_pars
                })
            else:
                self.saver.save_tensors({
                    'g_highs': self.g_highs, 
                    'g_lows': self.g_lows, 
                    's_highs': self.s_highs, 
                    's_lows': self.s_lows
                })
            
            # Clean up individual files after consolidation
            for file in g_highs_files + g_lows_files + s_highs_files + s_lows_files + c_par_files:
                os.remove(self.path + '/' + file)
        
        # Case 2: Consolidated tensor files exist
        elif all(os.path.exists(self.path + '/' + file) for file in files_check):
            if self.c_tightness:
                self.g_highs, self.g_lows, self.s_highs, self.s_lows, self.c_pars = \
                    self.tensor_loader("g_highs.pt"), \
                    self.tensor_loader("g_lows.pt"), \
                    self.tensor_loader("s_highs.pt"), \
                    self.tensor_loader("s_lows.pt"), \
                    self.tensor_loader("c_pars.pt")
            else:
                self.g_highs, self.g_lows, self.s_highs, self.s_lows = \
                    self.tensor_loader("g_highs.pt"), \
                    self.tensor_loader("g_lows.pt"), \
                    self.tensor_loader("s_highs.pt"), \
                    self.tensor_loader("s_lows.pt")
        
        # If conditional sampling is enabled, load/create the steady-state surrogate model
        if self.par_g_conditional_sampling:
            self.ss_surrogate = self.load_ss_surrogate_model() if self.load_ss_surrogate else self.ss_surrogate_model()
            self.ss_surrogate.eval()

    
    def g_sampler(self, N, std_size=2, upper_std=3):
        """Samples worker-firm match density values (g) that are perturbed random combinations of a steady state distribution.
        Generates N samples of gs where each sample is drawn from a truncated normal distribution centered between g_low and g_high.
        
        More specifically:
        g ~ TruncatedNormal(μ, σ²)
        where:
            μ = (g_high + g_low)/2
            σ = (g_high - g_low)/std_size
        with truncation at:
            lower bound: max(0, μ - upper_std*σ)
            upper bound: μ + upper_std*σ

        Args:
            N: Number of independent samples to generate
            std_size: Controls distribution spread relative to bounds.
                Higher values produce more concentrated samples.
                Default 2 means σ = (g_high-g_low)/2
            upper_std: Maximum allowed deviations from mean in standard deviations.

        Returns:
            tensor: Sampled densities shaped (N, nx*ny), where:
                - nx: Number of worker types
                - ny: Number of firm types
                - First dimension (N): Batch dimension
                - Second dimension (nx*ny): Flattened worker-firm densities g
        """
        # Get dimensions
        nx, ny = self.nx, self.ny  

        # Get density bounds (shape: (nx,ny) or (N,nx,ny))
        min_density = self.g_low  
        density_range = self.g_high_low_diff  # = g_high - g_low

        # Calculate distribution mean (midpoint between bounds)
        mean_density = min_density + density_range / 2  
        mean_density = mean_density.expand(N, nx, ny)

        # Calculate standard deviation (avoid too small values)
        std_dev = torch.clamp(density_range / std_size, min=torch.full_like(density_range, 1e-10))
        std_dev = std_dev.expand(N, nx, ny)

        # Sample from normal distribution
        density_samples = torch.normal(mean_density, std_dev)

        # Apply constraints to sampled densities
        max_density = mean_density + upper_std * std_dev
        density_samples = torch.clamp(density_samples, 
                                    min=torch.zeros_like(max_density),  # Density cannot be negative
                                    max=max_density)  # Upper bound at mean + upper_std*std_dev

        # Flatten worker-firm type dimensions
        return density_samples.flatten(start_dim=1)


    def sample(self, N, par=None):
        """
        Sample N training points (x,y,z,g) with current parameters.
        
        Returns:
            Tensor shaped (N, 4 + n_params) containing [x, y, z, g, params]
        """
        nz, device = self.nz, self.device
        par = self.par_draw if par is None else par  # Use existing params if none provided

        # Sample x and y coordinates from discrete distributions
        x = self.types_x[self.unif_x.multinomial(N, replacement=True)].reshape((N, 1))
        y = self.types_y[self.unif_y.multinomial(N, replacement=True)].reshape((N, 1))

        # Sample z uniformly and get g values
        z = torch.randint(nz, (N, 1), device=device)
        g = self.g_sampler(N)

        # Combine coords, g, and params
        return torch.cat([torch.hstack((x, y, z, g)), par.cat()], dim=-1)


    def draw_parameters(self, shape, device="cuda:0"):
        par_draw = self.range.sample(shape, device=device)
        return par_draw
    

    def y_init(self, X, par=None):
        """
        Generate initial guess for S(x,y,z) by interpolating from DSS values.
        Returns a tensor of shape (N,1).
        
        Args:
            X : torch.Tensor
                Input tensor of shape (N,3) containing (x,y,z) coordinates where we want initial guesses
            par : ParameterDict, optional
        """
        device, nx, ny, nz = self.device, self.nx, self.ny, self.nz
        par = self.par_draw if par is None else par  # Use drawn parameters if none provided
        
        # Extract coordinates from input tensor
        x_coords = X[:, 0].to(device)  # x coordinates
        y_coords = X[:, 1].to(device)  # y coordinates
        z_coords = X[:, 2].to(device)  # z coordinates
        
        # Get discrete state space surrogate values for current parameters
        gsc = self.ss_surrogate(par.cat())
        N = X.shape[0]  # Number of points
        
        # Extract S(x,y,z) values from surrogate
        if self.c_tightness:
            s_xyz = gsc[:, nx*ny*nz:-1]  # Exclude last column if c_tightness is True
        else:
            s_xyz = gsc[:, nx*ny*nz:]    # Include all columns otherwise
        
        # Reshape to (N, nx, ny, 2) where last dimension is for z=0 and z=1
        s_xyz = s_xyz.view(N, nx, ny, 2)
        
        # Convert continuous coordinates to discrete indices
        x_indices = (x_coords / (self.types_x[1] - self.types_x[0]) - 0.5).long()
        y_indices = (y_coords / (self.types_y[1] - self.types_y[0]) - 0.5).long()
        
        # Get surplus values at the computed indices
        init_guess = s_xyz[torch.arange(N), x_indices, y_indices, z_coords.int()]
        
        return init_guess


    def initial_guess(self, model, optimizer, scheduler, epochs, N=1000):
        """
        Train a model using DSS (deterministic steady state) values as initial guesses for S(x,y,z0)
        Uses DSS interpolated values as targets to pretrain/warm-start the model 
        before main training.
        
        Args:
            model : torch.nn.Module
                The model to be pretrained
            optimizer : torch.optim.Optimizer
                Optimizer for training
            scheduler : torch.optim.lr_scheduler
                Learning rate scheduler
            epochs : int
                Maximum number of training epochs
            N : int, optional
                Number of samples per epoch (default: 1000)
        """
        device = self.device
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Draw new parameters and update bounds
            self.par_draw = self.draw_parameters((N, 1), device)
            self._update_g_high_low(N)  # Update g_high and g_low based on current parameters
            
            # Sample points for training
            X_train = self.sample(N)
            
            # Prepare tensors for training
            optimizer.zero_grad()
            X_train = X_train.clone().requires_grad_(True).float()
            
            # Forward pass
            outputs = model(X_train)
            if self.c_tightness:
                gsc = self.ss_surrogate(self.par_draw.cat())
                c_par = gsc[:, -1].view(-1, 1).detach()
                self.par_draw.update({'c': c_par})
            
            # Get target values from DSS interpolation
            target_values = self.y_init(X_train).reshape(N, 1).to(device)
            
            # Compute/backpropagate loss
            loss = torch.mean(torch.square(outputs - target_values))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Get loss value for logging
            current_loss = loss.detach().cpu().numpy()
            
            # Periodic logging
            if epoch % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {epoch}: Total loss = {current_loss:.4e} at {elapsed:.2f}s")
                torch.save({
                    'modelS_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(self.path, f"model_S_init.pt"))
            
            # Check for convergence
            if current_loss < 1e-6:
                print(f'Converged at epoch {epoch} with training loss {current_loss}')
                self.par_draw = self.par  # Reset to original parameters
                break


    def training(self, pinn_S, model, optimizer, scheduler, epochs, save_path, option,
                save_freq=None, loss_threshold=1e-6, N=256, resample_epoch=2, 
                par_draw_epoch=100, rewrite_loss_file=True):
        """Train the model with the specified option ('S', 'V_u', or 'V_v').
        
        Args:
            pinn_S: The PINN model for S
            model: The model to train
            optimizer: Optimization algorithm
            scheduler: Learning rate scheduler
            epochs: Total training epochs
            save_path: Directory to save models
            option: Training mode ('S', 'V_u', or 'V_v')
            save_freq: Frequency to save model (in epochs)
            loss_threshold: Early stopping threshold
            N: Number of samples
            resample_epoch: Frequency to resample training points
            par_draw_epoch: Frequency to redraw parameters
            rewrite_loss_file: Whether to overwrite loss file
        """
        path, device = self.path, self.device
        start_time = time.time()
        
        # Initialize loss output file if needed
        if rewrite_loss_file:
            loss_file = os.path.join(self.path, f'output_loss_{option}.txt')
            open(loss_file, 'w').close()  # Clear file
        
        for epoch in range(1, epochs):
            # Draw new parameters
            if epoch % par_draw_epoch == 1: # use self.par before the first parameter draw
                self.par_draw = self.draw_parameters((N, 1), device)
                if self.c_tightness:
                    gsc = self.ss_surrogate(self.par_draw.cat())
                    c_par = gsc[:, -1].view(-1, 1).detach()
                    self.par_draw.update({'c': c_par})
                self._update_g_high_low(N)
            
            # Resample training points periodically
            if epoch % resample_epoch == 1: X_S = self.sample(N)
            
            # Calculate residuals based on training option
            # the pde loss: (pde_oper(S)-0)^2
            residuals, _, _, V, V_1 = self.S_pde_oper(model, X_S, self.par_draw)
            V_err = torch.mean(torch.abs(V - V_1)).item()  # Convert to scalar
            
            # Check for NaN values
            if np.isnan(V_err) and torch.any(torch.isnan(residuals)):
                print('NaN in loss and V_err')
                print(f"{torch.nonzero(torch.isnan(V)).squeeze().numel()} nan in V")
                print(f"{torch.nonzero(torch.isnan(V_1)).squeeze().numel()} nan in V_1")
                break
            
            # Optimization/backprop step
            loss = torch.mean(torch.square(residuals))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            ##### Logging and output
            current_time = time.time() - start_time
            
            # Save loss to file every 100 epochs
            if epoch % 100 == 0:
                with open(os.path.join(path, f'output_loss_{option}.txt'), 'a') as f:
                    f.write(f"Iter {epoch}: Loss = {loss:.4e} at {current_time:.2f}s\n")
                
                # During surplus training, also save vacancy error
                if option == 'S':
                    with open(os.path.join(path, f'V_err_{option}.txt'), 'a') as f:
                        f.write(f"Iter {epoch}: V_err = {V_err:.4e}\n")
            
            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                if option == 'S':
                    print(f"Epoch {epoch}, Loss_{option}: {loss.item():.5e}, "
                        f"V_err: {V_err}, time {current_time:.2f}s")
                else:
                    print(f"Epoch {epoch}, Loss_{option}: {loss.item():.5e}, "
                        f"time {current_time:.2f}s")
            
            # Save model periodically
            if save_freq and epoch % save_freq == 0:
                torch.save({
                    'modelS_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_path, f"model_S_epoch_{epoch}.pt"))
            
            # Early stopping if loss threshold reached
            if loss < loss_threshold:
                print(f"Epoch {epoch}, Loss: {loss.item()}, time {current_time:.2f}s")
                print("NN Converged.")
                break

    
    def mu_g(self, g, g_u, g_v, m_over_W_V, S, alphas, N, par=None):
        """
        Calculate the drift term for the joint density g.

        Args:
            g: Joint density tensor of shape (N, nx * ny).
            g_u: Marginal density for unemployed workers (N, nx).
            g_v: Marginal density for vacant jobs (N, ny).
            m_over_W_V: Matching efficiency over W and V (N, 1).
            S: Model output tensor of shape (N, nx, ny).
            alphas: Alpha values tensor of shape (N, nx, ny).
            N: Number of samples in batch.
            par: Parameters (optional). If None, uses self.par_draw.

        Returns:
            drift: Drift term for the joint density g (N, nx * ny).
        """

        nx, ny = self.nx, self.ny
        par = self.par_draw if par is None else par
        phi, delta, xi_e, eta = par.phi, par.delta, par.xi_e, par.eta

        # Prepare/reshape tensors
        alphas_flat = torch.flatten(alphas, start_dim=1)  # (N, nx, ny) -> (N, nx * ny)
        g_u_expanded = g_u.unsqueeze(2).expand(-1, -1, ny).flatten(start_dim=1)  # (N, nx * ny)
        g_v_expanded = g_v.unsqueeze(1).expand(-1, nx, -1).flatten(start_dim=1)  # (N, nx * ny)
        
        # Prepare S for pairwise comparisons
        S_flat = torch.flatten(S, start_dim=1)  # (N, nx * ny)
        S_flat_rep = S_flat.unsqueeze(1).expand(N, ny, nx * ny)  # (N, ny, nx * ny)
        S_x_y_tilde_flat = S.repeat_interleave(ny, dim=1).transpose(1, 2)  # (N, ny, nx * ny)

        # Calculate alpha terms
        #xi_e_reshaped = xi_e.view(-1, 1, 1)
        ## Compute alphas_p1 and alphas_p2
        alphas_p1 = self._compute_alpha_p(S_x_y_tilde_flat - S_flat_rep, S_x_y_tilde_flat, xi_e)
        alphas_p2 = self._compute_alpha_p(S_flat_rep - S_x_y_tilde_flat, S_flat_rep, xi_e)

        # Compute each term of the drift equation (see D.1 in the paper's appendix)
        # Term 1: -δg(x,y)
        # Decay due to exogenous separation with rate δ
        decay_term = -delta * g 
        
        # Term 2: -ηαᵇ(x,y)gₜ(x,y)
        # Endogenous separation
        separation_term = -eta * self.alpha_b_type * (1 - alphas_flat) * g 

        # Term 3: -Mₜᵉ∫αₜᵉ(x,y,ỹ)(gₜᵛ(ỹ)/Vₜ)gₜ(x,y)dỹ
        # Destruction from employed workers matching elsewhere
        destruction_term = -phi * m_over_W_V * g * torch.mean(alphas_p1 * g_v.reshape(N, ny, 1), dim=1)
        
        # Term 4: Mₜᵘαₜ(x,y)(gₜᵛ(y)/Vₜ)gₜᵘ(x) 
        # Creation from unemployed workers matching
        creation_term = m_over_W_V * alphas_flat * g_u_expanded * g_v_expanded 
        
        # Term 5: Mₜᵉ∫αₜᵉ(x,ỹ,y)(gₜᵛ(y)/Vₜ)gₜ(x,ỹ)dỹ
        # Reallocation from other already employed matches
        reallocation_term = phi * m_over_W_V * g_v_expanded * torch.mean(
            alphas_p2 * g.reshape(N, nx, ny).repeat_interleave(ny, dim=1).transpose(1, 2), dim=1)
        
        # Combine all terms
        drift = decay_term + destruction_term + creation_term + reallocation_term + separation_term
        return drift

    
    def calculate_alphas(self, model_S, z, g, par=None):
        """Calculate alphas (matching probabilities) between all firm-worker pairs.
        
        Args:
            model_S: Model that computes surplus values S(x,y)
            z: Aggregate state vector (N,)
            g: Joint density matrix (N, nx*ny)
            par: Optional parameters (uses self.par_draw if None)
            
        Returns:
            S: Computed surplus values (N, nx, ny)
            alphas: Matching probabilities (N, nx, ny)
        """
        device = self.device
        nx, ny = self.nx, self.ny
        par = self.par_draw if par is None else par
        N = len(z)  # Batch size
        
        # Prepare expanded tensors:
        ## Worker types (x) expanded to (N, nx, ny)
        worker_types = self.types_x.view(1, nx, 1).expand(N, nx, ny)
        ## Firm types (y) expanded to (N, nx, ny)
        firm_types = self.types_y.view(1, 1, ny).expand(N, nx, ny)
        ## Aggregate state (z) expanded to (N, nx, ny)
        z_broadcasted = z.view(N, 1, 1).expand(N, nx, ny)
        
        # Flatten all expanded tensors to (N*nx*ny, 1)
        input_worker = worker_types.reshape(-1, 1).to(device)
        input_firm = firm_types.reshape(-1, 1).to(device)
        input_z = z_broadcasted.reshape(-1, 1).to(device)
        
        # Prepare g matrix
        # From (N, nx*ny) to (N*nx*ny, nx*ny) (for pairwise alpha calculations)
        g_matrix = g.view(N, 1, -1).expand(N, nx*ny, nx*ny)
        g_matrix = g_matrix.reshape(N*nx*ny, -1).to(device)
        
        # Create full input tensor for model_S
        input_tensor_S = torch.cat([
            input_worker,       # Worker types
            input_firm,         # Firm types
            input_z,            # Aggregate state
            g_matrix,           # Joint density
            par.cat().expand(N, -1).unsqueeze(1).expand(N, nx*ny, -1).reshape(N*nx*ny, -1)  # Parameters
        ], dim=-1)
        
        # Compute surplus values and reshape 
        S = model_S(input_tensor_S).reshape(N, nx, ny)
        
        # Calculate matching probabilities (alphas)
        if self.alpha_type == 'continuous':
            alphas = torch.sigmoid(par.xi.view(-1, 1, 1) * S)
        else:  # discrete case
            alphas = (S >= 0).float()
        
        return S, alphas

    def marginals_U_V_E(self, g, S, alphas, par=None):
        """
        Calculate marginal distributions and aggregate statistics U, V, and E.

        Args:
            g: Joint density tensor of shape (N, nx * ny).
            S: Model output tensor of shape (N, nx, ny).
            alphas: Alpha values tensor of shape (N, nx, ny).
            par: Parameters (optional). If None, uses self.par_draw.

        Returns:
            g_e: Marginal density for employed workers (N, nx).
            g_u: Marginal density for unemployed workers (N, nx).
            U: Aggregate unemployment rate (N, 1).
            g_p: Marginal density for producing firms (N, ny).
            g_v: Marginal density for vacant firms (N, ny).
            V: Aggregate vacancy rate (N, 1).
            V_1: Aggregate vacancy rate under free entry (N, 1).
            g_f: Marginal density for firms (N, ny).
            E: Aggregate employment rate (N, 1).
        """

        # Extract parameters and device information
        device, nx, ny, g_w = self.device, self.nx, self.ny, self.gw
        par = self.par_draw if par is None else par
        rho, beta, kappa, nu, phi, xi_e, c = par.rho, par.beta, par.kappa, par.nu, par.phi, par.xi_e, par.c
        N = g.shape[0]

        # Reshape g from (N, nx * ny) to (N, nx, ny)
        g2d = g.reshape(N, nx, ny)

        # Calculate marginal densities and aggregate statistics
        g_e = torch.mean(g2d, dim=2)  # Marginal density for employed workers (N, nx)
        E = torch.mean(g_e, dim=1, keepdim=True)  # Aggregate employment rate (N, 1)
        g_u = g_w - g_e  # Marginal density for unemployed workers (N, nx)
        U = torch.mean(g_u, dim=1, keepdim=True)  # Aggregate unemployment rate (N, 1)
        W = U + phi * E

        g_p = torch.mean(g2d, dim=1)  # Marginal density for filled jobs (N, ny)
        P = torch.mean(g_p, dim=1, keepdim=True)  # Aggregate filled jobs rate (N, 1)

        # Calculate firm density and vacancy rate, if solving under free entry
        if self.no_fe:
            # If no firm entry, initialize g_f with default values
            g_f = torch.ones(ny, device=device, dtype=torch.float32) * self.init_gf
        else:
            A = (1 - beta) * torch.mean(alphas * S * g_u.repeat_interleave(ny).reshape(N, nx, ny), dim=(1, 2)).reshape(N, 1)

            # Expand S for pairwise comparisons
            S_expanded = S.unsqueeze(3).expand(N, nx, ny, ny)  # (N, nx, ny, ny)
            S_diff = S_expanded - S.unsqueeze(2)  # Pairwise differences (N, nx, ny, ny)

            # Calculate alphas_p
            alphas_p = torch.sqrt(1 / (1 + torch.exp(- xi_e.view(-1, 1, 1, 1) * S_diff)) /\
                                   (1 + torch.exp(- xi_e.view(-1, 1, 1, 1) * S_expanded)))  # (n,\tilde x,y,\tilde y)

            g_expanded = g2d.unsqueeze(2).expand(N, nx, ny, ny)  # Expand g for integration
            B = (1 - beta) * phi * (alphas_p * S_diff * g_expanded).mean(dim=(1, 2, 3)).unsqueeze(-1)

            # Calculate V_1 and g_f
            V_1 = (kappa / rho / c / W) ** (1 / nu) * W * (A + B) ** (1 / nu)
            g_f = (V_1.reshape(N, 1) + P).expand(N, ny)  # Marginal density for firms (N, ny)

        # Calculate marginal density for vacant jobs and aggregate vacancy rate
        g_v = g_f - g_p  # Marginal density for vacant jobs (N, ny)
        V = torch.mean(g_v, dim=1, keepdim=True)  # Aggregate vacancy rate (N, 1)

        # If no firm entry, set V_1 equal to V
        if self.no_fe: V_1 = V
        return g_e, g_u, U, g_p, g_v, V, V_1, g_f, E

        
    def S_pde_oper(self, model_S, X, par=None):
        """Compute the PDE residual for the surplus function S(x,y,z,g).
        
        Args:
            model_S: Neural network model for surplus S
            X: Batch input tensor containing N rows of values (x, y, z, g) 
                has shape: (N, 3 + nx*ny) where N is batch size
            par: Parameters (optional)
            
        Returns:
            S_pde_err: PDE residual
            alphas: Matching probabilities
            S_batch: Surplus values
            V: Aggregate vacancy rate
            V_1: Vacancy rate under free entry (just V if self.no_fe is True in config.yaml)
        """
        # Extract dimensions and parameters
        nx, ny = self.nx, self.ny
        types_x, types_y = self.types_x, self.types_y
        par = self.par_draw if par is None else par
        
        # Unpack parameters
        rho = par.rho
        beta = par.beta
        phi = par.phi
        delta = par.delta
        eta = par.eta
        xi_e = par.xi_e
        b = par.b
        
        # Extract input components
        x = X[:, 0].reshape(-1, 1)  # Worker types (N,1)
        y = X[:, 1].reshape(-1, 1)  # Firm types (N,1)
        z = X[:, 2].reshape(-1, 1)  # Productivity states (N,1)
        g = X[:, 3:(3 + nx*ny)]     # Density (N, nx*ny)
        N = len(x)                  # Batch size

        # Calculate surplus values from NN model
        S_batch = model_S(X)  # S(x,y,z,g) for current batch
        switched_z = ((z.int()) ^ 1).float()  # Switch z state (z')
        S_batch_switched_z = self._get_S_with_z(X, model_S, switched_z)  # S(x,y,z',g)
        
        # Calculate matching probabilities (alpha), and marginal distributions
        S, alphas = self.calculate_alphas(model_S, z, g, par)
        _, g_u, U, _, g_v, V, V_1, _, E = self.marginals_U_V_E(g, S, alphas, par)
        W = U + phi * E
        m_over_W_V = self.m(W, V) / (W * V)

        # Calculate gradient dS/dg
        Sg_pred = self._get_S_with_g(X, model_S, g)
        dS_dg = self.get_derivs_1order(Sg_pred, g)  # (N, nx*ny)
        g.requires_grad_(False)

        # Get indices, instead of type values, in batch elements
        indices_x = (x / (types_x[1] - types_x[0]) - 0.5).long().flatten()
        indices_y = (y / (types_y[1] - types_y[0]) - 0.5).long().flatten()

        # Select row or column of alpha matrix, for each batch element
        alphas_fixed_x = alphas[torch.arange(N), indices_x, :]  # (N, ny)
        alphas_fixed_y = alphas[torch.arange(N), :, indices_y]  # (N, nx)
        alphas_flat_b = self.alpha_b_type * (1 - alphas[torch.arange(N), indices_x, indices_y]).unsqueeze(1)


        # Calculating each line in surplus equation (see D.1 in the paper's appendix)
        # Term 1: -(ρ + δ)S(x, y, z, g) + F(x,y,z) - b - ηαᵇS(x, y, z, g)
        term1 = (-(rho + delta) * S_batch 
            + self.z_s[z.int()] * self.f_torch(x, y) 
            - b 
            - eta * alphas_flat_b * S_batch)

        # Term 2: -(1-β) * (m/WV) * (1/nₓ)∑ₖα(xₖ,y,z,g)*S(xₖ,y,z,g)*gᵘ(xₖ)
        term2 = -((1 - beta) * m_over_W_V * 
            torch.mean(alphas_fixed_y * g_u * S[range(N), :, indices_y], dim=1, keepdims=True))

        # Term 3: -(1-β) * φ* (m/WV) * (1/nₓnᵧ) * ∑ₖ∑ₗ αᵖ(y,xₖ,yₗ,z,g)ΔS g(xₖ,yₗ)
        # where ΔS = S(xₖ,y,z,g)-S(xₖ,yₗ,z,g)
        S_fixed_y = S[torch.arange(N), :, indices_y].unsqueeze(2)  # (N, nx, 1)
        S_diff = S_fixed_y - S  # (N, nx, ny)
        alphas_p = self._compute_alpha_p(S_diff, S_fixed_y, xi_e)
        term3 = -((1 - beta) * phi * m_over_W_V * 
            torch.mean(alphas_p * S_diff * g.reshape(N, nx, ny), dim=(1, 2)).reshape(N, 1))

        # Term 4: β * φ * (m/WV) * (1/nᵧ) * ∑ₗ αᵉ(x,y,yₗ,z,g) ΔS gᵛ(yₗ)
        # where ΔS = S(x,yₗ,z,g)-S(x,y,z,g), alpha^e(x,y,\tilde y) = S(x,\tilde y)>S(x,y)
        S_fixed_x = S[torch.arange(N), indices_x, :]  # (N, ny)
        S_diff_x = S_fixed_x - S_batch  # (N, ny)
        alpha_e = self._compute_alpha_e(S_diff_x, S_fixed_x, xi_e)
        term4 = beta * phi * m_over_W_V * torch.mean(alpha_e * S_diff_x * g_v, dim=1, keepdim=True)

        # Term 5: -β * (m/WV) * (1/n_y)∑ₗα(x,yₗ,z,g)S(x,yₗ,z,g)gᵛ(yₗ)
        term5 = -(beta * m_over_W_V * 
            torch.mean(alphas_fixed_x * g_v * S[range(N), indices_x, :], dim=1, keepdims=True))

        # Term 6: # Density evolution (μₖₗᵍ = g evolution drift)
        # ∑ₖ∑ₗ(∂S/∂gₖₗ)μᵍ(xₖ,yₗ)
        term6 = torch.sum(dS_dg * self.mu_g(g, g_u, g_v, m_over_W_V, S, alphas, N, par),
            axis=1, keepdims=True)

        # Term 7: Aggregate productivity switching component
        # λ(z)[S(x,y,z̃,g) - S(x,y,z,g)]
        term7 = self.lams[z.int()] * (S_batch_switched_z - S_batch)

        S_pde_err = term1 + term2 + term3 + term4 + term5 + term6 + term7
        return S_pde_err, alphas, S_batch, V, V_1

    def _compute_alpha_p(self, S_diff, S_ref, xi_e):
        """Calculation for computing alpha_p"""
        if self.alpha_type == 'continuous':
            if self.alpha_e_type == 'D':
                return torch.sqrt(1 / (1 + torch.exp(-xi_e.view(-1, 1, 1) * S_diff)) / (1 + torch.exp(-xi_e.view(-1, 1, 1) * S_ref)))
            else:
                return 1 / (1 + torch.exp(-xi_e.view(-1, 1, 1) * S_diff))
        else:  # discrete case
            if self.alpha_e_type == 'D':
                return (S_diff >= 0) * (S_ref >= 0)
            else:
                return S_diff >= 0
            
    def _compute_alpha_e(self, S_diff_x, S_fixed_x, xi_e):
        """Calculation for computing alpha_e"""
        if self.alpha_type == 'continuous':
            if self.alpha_e_type == 'D':
                return torch.sqrt(
                    1 / (1 + torch.exp(-xi_e * S_diff_x)) / (1 + torch.exp(-xi_e * S_fixed_x)))  # ->(N,ny)
            else:
                return  1 / (1 + torch.exp(-xi_e * S_diff_x))
        elif self.alpha_type == 'discrete':
            if self.alpha_e_type == 'D':
                return  (S_diff_x >= 0) * (S_fixed_x >= 0)  # ->(N,ny)
            else:
                return  S_diff_x >= 0
    

    def _update_g_high_low(self, N, par=None):
        """
        Update g_high and g_low based on current par_draw. 
        Since par_draw is not updated in each epoch, we can save time by updating g_high and g_low only when par_draw changes.
        """
        par = self.par_draw if par is None else par
        if self.par_g_conditional_sampling:
            with torch.no_grad():
                gsc = self.ss_surrogate(par.cat())
            g_xyz = gsc[:, :self.nx*self.ny*self.nz].view(N, self.nx, self.ny, self.nz)  # (N,nx*ny*nz)->(N,nx,ny,nz), not M!
            self.g_low, g_high = torch.min(g_xyz, dim=3).values, torch.max(g_xyz, dim=3).values  # (N,nx,ny)
        else:
            g_high = torch.max(self.g_highs, dim=0).values  # (nx, ny)
            self.g_low = torch.min(self.g_lows, dim=0).values  # (nx, ny)
        self.g_high_low_diff = g_high - self.g_low  # (nx, ny) or (N,nx,ny)
            

    # Helper functions for automatic differentiation
    def get_derivs_1order(self, y_pred, x):
            """ Returns first order derivatives.
                Uses automatic differentiation.
            """
            dy_dx = torch.autograd.grad(y_pred, x,
                                        create_graph=True,
                                        grad_outputs=torch.ones_like(y_pred))[0]
            return dy_dx  # Return 'automatic' gradient.
    
    def _get_S_with_g(self, X, model_S, g):
        """Compute S with replaced g values"""
        nx, ny = self.nx, self.ny
        g.requires_grad_(True)  # Initiate auto. diff. tracking
        X_temp = X.clone()
        X_temp[:, 3:(3 + nx*ny)] = g
        return model_S(X_temp)

    def _get_S_with_z(self, X, model_S, new_z):
        """Compute S with replaced z values"""
        X_temp = X.clone()
        X_temp[:, 2:3] = new_z
        return model_S(X_temp)