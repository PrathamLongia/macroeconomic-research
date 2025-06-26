from omegaconf import DictConfig, OmegaConf
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from structures import Parameters
from train_nn import Train_NN, Master_PINN_S
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from helpers import TimeSeries
#from plot_helpers import to_percent

class Simulation:
    def __init__(self, ct: Train_NN, pinn_S: Master_PINN_S, par: Parameters, 
                 pinn_V_u=None, pinn_V_v=None, cfg: DictConfig = None):
        """
        Initialize simulation class with configuration, NN models, and parameters.
        """
        self.ct = ct
        self.pinn_S = pinn_S
        self.par = par
        self.pinn_V_u = pinn_V_u
        self.pinn_V_v = pinn_V_v
        self.cfg = cfg
        self.device = ct.device
        self.dtype = getattr(torch, getattr(ct, 'dtype', 'float32'), torch.float32)
        self.nx = ct.nx
        self.ny = ct.ny
        self.z_s = ct.z_s

    def _initialize_tensors(self, N, T, nx, ny, dtype):
        """
        Helper function to create placeholder tensors for simulation results.
        """
        return {
            'alpha_series': torch.empty((N, T-1, nx, ny), dtype=dtype, device=self.device),
            'g_series': torch.empty((N, T-1, nx, ny), dtype=dtype, device=self.device),
            'U_series': torch.empty((N, T-1), dtype=dtype, device=self.device),
            'V_series': torch.empty((N, T-1), dtype=dtype, device=self.device),
            'TFP_series': torch.empty((N, T-1), dtype=dtype, device=self.device),
            'e2e': torch.empty((N, T-1), dtype=dtype, device=self.device),
            'u2e': torch.empty((N, T-1), dtype=dtype, device=self.device),
            'e2u': torch.empty((N, T-1), dtype=dtype, device=self.device)
        }

    def simulate_exogenous(self, T: int, N: int = 1, dt: float = 0.01, initial_state=None):
        """
        Simulates an aggregate productivity state transition process (two state Markov chain).
        """
        if initial_state is None: 
                initial_state = torch.randint(0, 2, (N,), device=self.device)
                
        P = torch.tensor([
            [1 - self.ct.lams[0] * dt, self.ct.lams[0] * dt],
            [self.ct.lams[1] * dt, 1 - self.ct.lams[1] * dt]
        ], device=self.device)

        z_t = torch.zeros(N, T, dtype=torch.int64, device=self.device)
        z_t[:, 0] = initial_state

        for t in range(1, T):
            next_state = torch.multinomial(P[z_t[:, t - 1]], 1)
            z_t[:, t] = next_state.squeeze(1)
        return z_t


    def simulate_endogenous(self, dt=0.01, T=2000, N=1, z_t=None, 
                            initial_condition=None, alpha0=None, U0=None, 
                            V0=None, ergodicT=None, t_wage_snap=None, printer=True,
                            tfp_denom=True, TFP0=None, if_mean_wage=False, 
                            e2e0=None, u2e0=None, e2u0=None, PE_simulation=False):
        """
        Simulate endogenous variables (g,alpha,U,V, etc.) based on exogenous states (z_t).
        """
        nx, ny = self.nx, self.ny
        device, dtype = self.device, self.dtype

        # Initialize exogenuos parameters and initial state
        N_par = 1
        if self.par.__getattribute__(self.par.keys()[0]).dim() != 1:
            N_par = self.par.shape[0]
            self.par = self.par.repeat_interleave(self.ct.N_erg, dim=0)

        if initial_condition is None:
            g_dss = np.load(os.path.join(self.ct.path, 'g_ss.npy'))
            initial_condition = torch.tensor(g_dss, dtype=dtype, device=device)
            initial_condition = initial_condition.expand(N, nx, ny)
            
        z_t = z_t if z_t is not None else self.simulate_exogenous(T, N, dt)
        T = z_t.size(1)

        # Initialize placeholder tensors for simulation results
        tensors = self._initialize_tensors(N, T, nx, ny, dtype)
        #alpha_series = tensors['alpha_series']
        #g_series = tensors['g_series']
        if ergodicT is None:
            tensors = self._initialize_tensors(N, T, nx, ny, dtype)
            alpha_series = tensors['alpha_series']
            g_series = tensors['g_series']
        else:
            if N_par > 1:
                alpha_series = torch.zeros((N_par, nx, ny), dtype=dtype, device=device)
                g_series = torch.zeros((N_par, nx, ny), dtype=dtype, device=device)
            else:
                alpha_series = torch.zeros((nx, ny), dtype=torch.float64, device=device)
                g_series = torch.zeros((nx, ny), dtype=torch.float64, device=device)
        U_series = tensors['U_series']
        V_series = tensors['V_series']
        TFP_series = tensors['TFP_series']
        e2e = tensors['e2e']
        u2e = tensors['u2e']
        e2u = tensors['e2u']

        g_next = initial_condition
        x_tensor = self.ct.types_x.view(1, nx, 1).expand(N, nx, ny).flatten().unsqueeze(-1)
        y_tensor = self.ct.types_y.view(1, 1, ny).expand(N, nx, ny).flatten().unsqueeze(-1)
        f_tensor = self.ct.f_torch(x_tensor, y_tensor)

        # Main simulation loop
        for t in range(T - 1):
            if t % int(T / 2) == 0 and printer:
                print(f"Simulating period {t}/{T}...")

            g_prev = g_next
            z_tensor = self._prepare_z_tensor(z_t, t, N, nx, ny)
            TFP_series[:, t] = self._calculate_tfp(z_tensor, f_tensor, g_prev, tfp_denom)

            # Calculate Surplus and alpha using NN model, pinn_S
            X_S = self._prepare_inputs(x_tensor, y_tensor, z_tensor, g_prev, PE_simulation)
            S_val = self.pinn_S(X_S).detach()
            surplus_values = S_val.reshape(N, nx, ny)
            alpha_values = 1 / (1 + torch.exp(-self.par.xi.view(-1, 1, 1) * surplus_values))

            # Update series
            if ergodicT is None:
                g_series[:, t, :, :] = g_next
            elif t > (T - 2 - ergodicT):
                self._update_ergodic_series(g_series, g_next, N_par, ergodicT)

            # Economic calculations
            ge = torch.mean(g_prev, dim=2)
            gp = torch.mean(g_prev, dim=1)
            E_t = torch.mean(ge, dim=1, keepdim=True)
            U_t = torch.mean(self.ct.gw) - E_t
            gu_3d = self._calculate_gu(ge, N)
            W_t = U_t + self.par.phi * E_t

            # Update V_t and density of firm types
            if self.ct.no_fe:
                gf = torch.ones(N, ny, device=device, dtype=dtype) * self.ct.init_gf
                V_t = torch.mean(gv, axis=1, keepdim=True)  # (N,1)
            else: 
                V_t, gf = self._update_Vt_gf(alpha_values, surplus_values, gu_3d, 
                                                  g_prev, W_t, E_t, N)
            gv = gf - gp

            # Store results
            U_series[:, t] = U_t.squeeze(1)
            V_series[:, t] = V_t.squeeze(1)
            m_val = self.ct.m(W_t, V_t)

            # Calculate flow rates
            term1, term2, term3, term4 = self._calculate_flow_rates(g_prev, alpha_values, surplus_values, 
                                            gu_3d, gv, m_val, W_t, V_t)
            mu_g = term1 + term2 + term3 + term4
            g_next = g_prev + dt * mu_g

            # Calculate and store moments
            e2e[:, t] = torch.mean(-term2, dim=(1, 2))
            u2e[:, t] = torch.mean(term3, dim=(1, 2))
            e2u[:, t] = torch.mean(-term1, dim=(1, 2))
            e2e_2 = torch.mean(term4, dim=(1, 2))
            if self.ct.flow_rate:
                e2e_2 = e2e_2 / E_t.squeeze() 
                e2e[:, t] = e2e[:, t] / E_t.squeeze() 
                e2u[:, t] = e2u[:, t] / E_t.squeeze() 
                u2e[:, t] = u2e[:, t] / U_t.squeeze() 
            if self.ct.monthly_flow:
                e2e_2 = e2e_2 / 12
                e2e[:, t] = e2e[:, t] / 12
                e2u[:, t] = e2u[:, t] / 12
                u2e[:, t] = u2e[:, t] / 12

            # Stop immediately if any results are nan
            if torch.any(torch.isnan(g_next)):
                print(f'NaN detected at t={t}')
                break

        
        result = self._finalize_results(
            g_series, alpha_series, U_series, V_series, TFP_series,
            e2e, u2e, e2u, initial_condition, alpha0, U0, V0, TFP0,
            e2e0, u2e0, e2u0, ergodicT, N, tfp_denom
        )
        np.save("alpha_erg.npy", alpha_values.detach().cpu().numpy()[-1])
        #np.save("S_erg.npy", alpha_values.detach().cpu().numpy())
        np.save("g_erg.npy", g_next.detach().cpu().numpy()[-1])
        np.save("S_erg.npy", surplus_values.detach().cpu().numpy()[-1])

        # No wage dynamics for now....
        #if self.ct.wage_dynamics and self.pinn_V_u is not None and self.pinn_V_v is not None:
        #    result = self._handle_wage_dynamics(result, ergodicT, if_mean_wage)

        return result

    def ergodics(self, dt=0.01, T=3001, N_erg=1,
                              if_plot=False, if_save=True, ergodicT=None, 
                              if_mean_wage=None, printer=True):
        """
        Run full ergodic simulation.
        """
        
        if_mean_wage = ('mean_wage' in self.cfg.moment_names) if if_mean_wage is None else if_mean_wage
        N_par = self.par.shape[0]
        N = N_par * N_erg
        z_t = self.simulate_exogenous(T, N, dt)
        ergodicT = max(round(0.9 * T), 100) if ergodicT is None else ergodicT

        # Load steady state values
        alpha_ss, U_ss, V_ss, TFP_ss, e2e_ss, u2e_ss, e2u_ss = self._load_steady_state_values()

        # Run simulation
        result = self.simulate_endogenous(
            dt=dt, T=T, N=N, z_t=z_t, ergodicT=ergodicT,
            if_mean_wage=if_mean_wage, U0=U_ss, V0=V_ss, 
            e2e0=e2e_ss, u2e0=u2e_ss, e2u0=e2u_ss, 
            TFP0=TFP_ss.expand(N, 1), alpha0=alpha_ss, printer=printer
        )
        g_series, alpha_series, U_series, V_series, TFP_series, e2e, u2e, e2u = result

        #if if_plot: self._plot_ergodic_simulation(result, dt)
        #if if_save: self._save_simulation_results(g_series, alpha_series)

        moment_series = [U_series, V_series, e2e, e2u, u2e, TFP_series]
        ts = TimeSeries()
        sim_moments = ts.mean_std_autocorr(moment_series, T-ergodicT, None, self.cfg.moment_names, N_par=N_par)

        torch.cuda.empty_cache()
        return sim_moments
    

    def _load_steady_state_values(self):
        '''
        Load steady-state values from existing files.
        '''
        ss_files = [
            'alpha_ss.npy', 
            'U_ss.npy', 
            'V_ss.npy', 
            'TFP_ss.npy',
            'e2e_ss.npy',
            'u2e_ss.npy',
            'e2u_ss.npy'
        ]
        
        loaded = []
        for fname in ss_files:
            tensor = self.ct.tensor_loader(fname)
            loaded.append(tensor)
            
        return (
            loaded[0],  # alpha_ss
            loaded[1],  # U_ss
            loaded[2],  # V_ss
            loaded[3],  # TFP_ss
            loaded[4],  # e2e_ss
            loaded[5],  # u2e_ss
            loaded[6]   # e2u_ss
        )
    

    def _plot_ergodic_simulation(self, result, dt):
        '''
        Helper function to handle plotting simulation results.
        '''
        from plot import to_percent
        metrics = ['U_erg', 'V_erg', 'TFP_erg', 'e2e_erg', 'u2e_erg', 'e2u_erg']
        ergodic_values = {metric: torch.mean(val, dim=0) for metric, val in zip(metrics, result)}
        x_values = np.arange(len(next(iter(ergodic_values.values()))[:-1])) * dt

        for key in ergodic_values:
            plt.figure(figsize=(10, 6))
            y_data = ergodic_values[key][1:-1].cpu().numpy()
            if key == 'TFP_erg':
                y_data = (y_data / ergodic_values[key][0].cpu().numpy()) - 1
            plt.plot(x_values[1:], y_data)
            plt.title(f'Ergodic {key.split("_")[0]}')
            plt.xlabel('Years')
            plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
            plt.savefig(os.path.join(self.ct.path, f'ergSim_{key.split("_")[0]}.png'))
            plt.close()

    def _save_simulation_results(self, g_series, alpha_series):
        '''
        Helper function to handle saving simulation results as .npy files.
        '''     
        #components = ['g_erg', 'alpha_erg']
        np.save(os.path.join(self.ct.path, 'g_erg.npy'), g_series.detach().cpu().numpy())
        np.save(os.path.join(self.ct.path, 'alpha_erg.npy'), alpha_series.detach().cpu().numpy())


    def _finalize_results(self, g_series, alpha_series, U_series, V_series,
                        TFP_series, e2e, u2e, e2u, initial_condition,
                        alpha0, U0, V0, TFP0, e2e0, u2e0, e2u0,
                        ergodicT, N, tfp_denom):
        """
        Add pre-shock period and format final results.
        """
        nx, ny = self.nx, self.ny
        dtype = getattr(torch, getattr(self.ct, 'dtype', 'float32'), torch.float32)
        x_tensor = self.ct.types_x.view(1, nx, 1).expand(N, nx, ny).flatten().unsqueeze(-1)  # (N*nx*ny,1)
        y_tensor = self.ct.types_y.view(1, 1, ny).expand(N, nx, ny).flatten().unsqueeze(-1)
        f_tensor = self.ct.f_torch(x_tensor, y_tensor)  # (N*nx*ny,1)

        if ergodicT is None:
            g_series = torch.cat((initial_condition.view(N, 1, nx, ny), g_series), dim=1)  # (N,T-1,nx,ny)->(N,T,nx,ny)
            alpha_series = torch.cat((alpha0.view(1, 1, nx, ny).expand(N, 1, nx, ny), alpha_series), dim=1)  # (N,T-1,nx,ny)->(N,T,nx,ny)
        else:
            g_series = g_series.to(dtype=dtype)
            alpha_series = alpha_series.to(dtype=dtype)

        U_series = torch.cat((U0.view(1, 1).expand(N, 1), U_series), dim=1)  # (N,T-1)->(N,T)
        V_series = torch.cat((V0.view(1, 1).expand(N, 1), V_series), dim=1)  # (N,T-1)->(N,T)

        if TFP0 is None:
            TFP0 = (self.par.z_0 * f_tensor * initial_condition.view(N * nx * ny, 1)).view(N, nx, ny).mean(
                dim=(1, 2))
            if tfp_denom:
                TFP0 = TFP0 / initial_condition.mean(dim=(1, 2))
            TFP0 = TFP0.unsqueeze(-1)
        TFP_series = torch.cat((TFP0, TFP_series), dim=1)  # (N,T-1)->(N,T)

        e2e = torch.cat((e2e0.view(1, 1).expand(N, 1), e2e), dim=1)
        u2e = torch.cat((u2e0.view(1, 1).expand(N, 1), u2e), dim=1)
        e2u = torch.cat((e2u0.view(1, 1).expand(N, 1), e2u), dim=1)
        result = (g_series, alpha_series, U_series, V_series, TFP_series, e2e, u2e, e2u)
        return result



    def _calculate_flow_rates(self, g_prev, alpha_values, surplus_values, 
                         gu_3d, gv, m_val, W_t, V_t):
        '''
        Calculate mu_g
        '''

        N, nx, ny = g_prev.shape[0], self.nx, self.ny

        # Reshape parameters for broadcasting
        delta = self.par.delta.view(-1, 1, 1)
        eta = self.par.eta.view(-1, 1, 1)
        phi = self.par.phi.view(-1, 1, 1)
        
        # Term 1
        alpha_b = self.ct.alpha_b_type * (1 - alpha_values)
        term1 = -delta * g_prev - eta * alpha_b * g_prev

        # Term 2
        m_over_W_V = (m_val / W_t / V_t).unsqueeze(2).expand(N, self.nx, self.ny)
        S_flat = torch.flatten(surplus_values, start_dim=1)  # (N,nx,ny)->(N,nx*ny). (n,x,y)
        S_flat_rep = S_flat.unsqueeze(1).expand(N, self.ny,
                                                self.nx * self.ny)  # (N,nx*ny)->(N,1,nx*ny)->(N,ny,nx*ny). (n,\tilde y,(x,y)). The same y on dim=1 for each (n,x,y)
        S_x_y_tilde_flat = surplus_values.repeat_interleave(self.ny,
                                                            dim=1)  # (N,nx,ny)->(N,nx*ny,ny). (n, (x,y), \tilde y). All types of \tilde y for each (n,(x,y))
        S_x_y_tilde_flat = S_x_y_tilde_flat.transpose(1, 2)  # (N,nx*ny,ny)->(N,ny,nx*ny). (n, \tilde y, x,y)
        if self.ct.alpha_type == 'continuous':
            if self.ct.alpha_e_type == 'D':
                alphas_p1 = torch.sqrt(1 / (1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * (S_x_y_tilde_flat - S_flat_rep))) / (
                        1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * S_x_y_tilde_flat)))  # (N,ny,nx*ny).
            else:
                alphas_p1 = 1 / (1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * (S_x_y_tilde_flat - S_flat_rep)))  # (N,ny,nx*ny).
        elif self.ct.alpha_type == 'discrete':
            if self.ct.alpha_e_type == 'D':
                alphas_p1 = S_x_y_tilde_flat >= S_flat_rep and S_x_y_tilde_flat >= 0
            else:
                alphas_p1 = S_x_y_tilde_flat >= S_flat_rep
        
        term2 = -phi * m_over_W_V * g_prev * torch.mean(
            alphas_p1 * gv.reshape(N, self.ny, 1), dim=1).view(-1, self.nx, self.ny)

        # Term 3
        term3 = m_over_W_V * alpha_values * gu_3d * gv.view(-1, 1, self.ny)

        # Term 4
        if self.ct.alpha_e_type == 'D':
            alphas_p2 = torch.sqrt(1/(1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * (S_flat_rep - S_x_y_tilde_flat))) / 
                        (1 + torch.exp(-self.par.xi_e.view(-1, 1, 1)  * S_flat_rep)))
        else:
            alphas_p2 = 1/(1 + torch.exp(-self.par.xi_e.view(-1, 1, 1)  * (S_flat_rep - S_x_y_tilde_flat)))
        
        if self.ct.alpha_type == 'continuous':
            if self.ct.alpha_e_type == 'D':
                alphas_p2 = torch.sqrt(1 / (1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * (S_flat_rep - S_x_y_tilde_flat))) / (
                        1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * S_flat_rep)))  # (N,ny,nx*ny).
            else:
                alphas_p2 = 1 / (1 + torch.exp(-self.par.xi_e.view(-1, 1, 1) * (S_flat_rep - S_x_y_tilde_flat)))  # (N,ny,nx*ny).
        elif self.ct.alpha_type == 'discrete':
            if self.ct.alpha_e_type == 'D':
                alphas_p2 = S_x_y_tilde_flat >= S_flat_rep and S_flat_rep >= 0
            else:
                alphas_p2 = S_flat_rep >= S_x_y_tilde_flat
        term4 = self.par.phi.view(-1, 1, 1) * m_over_W_V * gv.view(-1, 1, self.ny) * torch.mean(
            alphas_p2 * g_prev.reshape(N, self.nx, self.ny).repeat_interleave(self.ny, dim=1).transpose(1, 2), dim=1).reshape(
            N, self.nx, self.ny)

        return term1, term2, term3, term4

    def _prepare_z_tensor(self, z_t, t, N, nx, ny):
        '''
        Reshape z: ->(N,1,1)->(N,nx*ny,1)->(N*nx*ny,1)
        '''
        return z_t[:, t].reshape(N, 1, 1).expand(N, nx * ny, 1).reshape(N * nx * ny, 1)
    
    def _update_ergodic_series(self, series, values, N_par, ergodicT):
        '''
        Update ergodic series with averaged values.
        '''
        if N_par > 1:
            series += values.view(N_par, self.ct.N_erg, self.nx, self.ny).mean(dim=1) /  ergodicT
        else:
            series += values.mean(dim=0) /ergodicT

    def _calculate_gu(self, ge, N):
        """
        Calculate 3D unemployment distribution tensor, with shape (N,nx,ny)
        """
        return (self.ct.gw.reshape(self.nx, 1) - ge.unsqueeze(2)).expand(-1, -1, self.ny)
    
    def _prepare_inputs(self, x_tensor, y_tensor, z_tensor, g_prev, PE_simulation):
        '''
        Prepare input tensor for NN forward pass.
        '''
        N, nx, ny = g_prev.shape[0], self.nx, self.ny
        
        # Handle g_prev formatting based on simulation type
        if PE_simulation:
            g_flat = g_prev.view(N, nx * ny).unsqueeze(1)
            g_flat = g_flat.expand(N, nx * ny, nx * ny).reshape(N * nx * ny, nx * ny)
        else:
            g_flat = g_prev.reshape(N, nx * ny).unsqueeze(1)
            g_flat = g_flat.expand(N, nx * ny, nx * ny).reshape(N * nx * ny, nx * ny)

        # Concatenate
        X_S = torch.cat([
            x_tensor,          # Worker types (N*nx*ny, 1)
            y_tensor,          # Firm types (N*nx*ny, 1)
            z_tensor,          # Productivity state (N*nx*ny, 1)
            g_flat             # Previous distribution (N*nx*ny, nx*ny)
        ], dim=1)

        # Add parameters
        if self.par.shape[0] == 1:  # Single parameter set
            X_S = torch.cat([X_S, self.par.cat().expand(X_S.shape[0], -1)], dim=-1)
        else:  # Multiple parameter sets
            X_S = torch.cat([
                X_S,
                self.par.cat().repeat_interleave(nx * ny, dim=0)
            ], dim=-1)

        return X_S


    def _calculate_tfp(self, z_tensor, f_tensor, g_prev, tfp_denom):
        tfp = (self.z_s[z_tensor.int()] * f_tensor * g_prev.flatten().unsqueeze(1))
        tfp = tfp.reshape(g_prev.shape).mean(dim=(1, 2))
        return tfp / g_prev.mean(dim=(1, 2)) if tfp_denom else tfp


    def _update_Vt_gf(self, alpha_values, surplus_values, gu_3d, g_prev, W_t, E_t, N):
        A = (1 - self.par.beta) * torch.mean(alpha_values * surplus_values * gu_3d, dim=(1, 2)).unsqueeze(1)
        S_expanded = surplus_values.unsqueeze(3).expand(N, self.nx, self.ny,
                                self.ny)  # (N,nx,ny)->(N,nx,ny,1)->(N,nx,ny,ny), the same on dim=3, \tilde y
        S_diff = S_expanded - surplus_values.unsqueeze(
                2)  # surplus_values.unsqueeze(2):(N,nx,1,ny), all types of \tilde y on dim=3, the same y on dim=2
        #S_diff = self._calculate_s_diff(surplus_values)
        if self.ct.alpha_e_type == 'D':
                alphas_p = torch.sqrt(
                    1 / (1 + torch.exp(- self.par.xi_e.view(-1, 1, 1, 1) * S_diff)) / \
                        (1 + torch.exp(- self.par.xi_e.view(-1, 1, 1, 1) * S_expanded)))  # (n,\tilde x,y,\tilde y)
        else:
            alphas_p = 1 / (1 + torch.exp(- self.par.xi_e.view(-1, 1, 1, 1) * S_diff))  # (n,\tilde x,y,\tilde y)
        B = (1 - self.par.beta) * self.par.phi * (alphas_p * S_diff * g_prev.unsqueeze(2)).mean(dim=(1, 2, 3)).unsqueeze(1)
        V_t = (self.par.kappa / self.par.rho / self.par.c / W_t) ** (1/self.par.nu) * W_t * (A + B) ** (1/self.par.nu)
        gf = (V_t.reshape(N, 1) + E_t).expand(N, self.ny)
        return V_t, gf