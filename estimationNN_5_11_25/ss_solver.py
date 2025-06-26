import os
import sys
import numpy as np
from numpy import linalg as LA
from env import Environment
import multiprocessing
import os
import sys
import numpy as np
from numpy import linalg as LA
import time
from structures import Parameters

class SteadyStateSolver(Environment):
    """Subclass for handling steady state solutions with/without free entry 
    using fixed-point iteration"""
    def __init__(self, **kwargs):
        # Initialize the parent class with kwargs
        super().__init__(**kwargs)

        # letting gf all 1 would imply U=V when z=z_0: V+P=gf=gp+gv,gw=U+E,mean(gp)=P=E=mean(ge)
        self.init_gf = 1 + self.U_target * (self.init_v_u_ratio - 1)
        
        # Initialize variables
        self.g = None
        self.surplus = None
        self.alphas = None
        self.U = None
        self.V = None

    def marginals_U_V(self, g, g_f=None, g_v=None):
        """Calculate marginal distributions from joint worker-firm matching density g"""
        g_e = np.mean(g, axis=1)
        g_u = self.gw_np - g_e
        U = np.mean(g_u)
        g_p = np.mean(g, axis=0)
        if (g_f is not None) and (g_v is None):
            g_v = g_f - g_p
        V = np.mean(g_v)
        E = np.mean(g_e)
        return g_e, g_u, U, g_p, g_v, V, E

    def create_g_no_free_entry(self, z, g_f=None, g_v=None, par=None, g=None, surplus=None, payoffs=None, b=None, alphas_e=None, suffix='', printer=False, save_c=False):
        """Solve the fixed point steady state s&g for general beta, only g for block-recursive models. """
        nx, ny, g_w = self.nx, self.ny, self.gw_np
        par = self.par_draw if par is None else par
        xi, xi_e, delta, rho, kappa, beta, phi, eta = par.xi, par.xi_e, par.delta, par.rho, par.kappa, par.beta, par.phi, par.eta
        if b is None:
            b = par.b
        if isinstance(b, np.ndarray):
            if b.shape == (nx,):
                b = b[:, None]  # (nx,)->(nx,ny)
        tol = 1e-14  # Absolute tolerance level for the fixed-point iteration

        surplus = np.random.rand(nx, ny) * 0.9 - 0.5 if surplus is None else surplus  # initialize S(x,y) to (-0.4,0.8)
        if g is None:
            g = np.random.uniform(low=0.04, high=.6, size=(nx, ny))
        if self.alpha_type == 'continuous':
            alphas = 1 / (1 + np.exp(-xi * surplus))
        elif self.alpha_type == 'binary':
            alphas = surplus >= 0
        alphas_b = self.alpha_b_type * (1 - alphas)  # alpha^b=1 if surplus<0. Endogenous separation
        if alphas_e is None:
            s_diff = surplus[:, None, :] - surplus[:, :, None]
            if self.alpha_type == 'continuous':
                if self.alpha_e_type == 'D':
                    alphas_e = np.sqrt(1 / (1 + np.exp(-xi_e * s_diff)) / (1 + np.exp(-xi_e * surplus[:, None, :])))
                else:
                    alphas_e = 1 / (1 + np.exp(-xi_e * s_diff))
            elif self.alpha_type == 'binary':
                if self.alpha_e_type == 'D':
                    alphas_e = (s_diff >= 0) * (surplus[:, None, :] >= 0)
                else:
                    alphas_e = s_diff >= 0
        if payoffs is None:
            types_x = np.linspace(1 / nx / 2, 1 - 1 / nx / 2, nx)
            types_y = np.linspace(1 / ny / 2, 1 - 1 / ny / 2, ny)
            payoffs = z * self.calc_f(np.expand_dims(types_x, axis=1), np.expand_dims(types_y, axis=0))  # also what LR claim in their text

        def marginals_U_V(g, g_v=None):
            g_e = np.mean(g, axis=1)
            g_u = g_w - g_e
            U = np.mean(g_u, axis=0)
            g_p = np.mean(g, axis=0)
            if (g_f is not None) and (g_v is None):
                g_v = g_f - g_p
            V = np.mean(g_v, axis=0)
            E = np.mean(g_e)  # ==P
            return g_e, g_u, U, g_p, g_v, V, E

        distance = sys.float_info.max
        start = time.time()
        lr = 0.02  # learning rate in fixed-point algorithm for each new iteration, both g and surplus
        while distance >= tol:
            alphas_b = (1 - alphas) * self.alpha_b_type  # alpha^b=1 if surplus<0. Endogenous separation
            g_e, g_u, U, g_p, g_v, V, E = marginals_U_V(g)
            W = U + phi * E
            g_prev = g.copy()

            # alpha^e(x,\tilde y, y),
            # SS KFE numerator integrates \tilde y, so axis=1
            # alpha^e(x,\tilde y,y):=S(x,y)>S(x,\tilde y) &　S(x,y)>0=:alpha^p(y, x,\tilde y), so we can just use alpha^e.
            # \alpha^e[i,j,k]=surplus[i,k] > surplus[i,j] & surplus[i,k]>0
            s_diff = surplus[:, None, :] - surplus[:, :, None]
            # equivalent to surplus.repeat_interleave(ny, dim=0).reshape(nx, ny, ny) - surplus.reshape(nx,ny,1).expand(nx, ny, ny) if tensor
            if self.alpha_type == 'continuous':
                if self.alpha_e_type == 'D':
                    alphas_e = np.sqrt(1 / (1 + np.exp(-xi_e * s_diff)) / (1 + np.exp(-xi_e * surplus[:, None, :])))
                else:
                    alphas_e = 1 / (1 + np.exp(-xi_e * s_diff))
            elif self.alpha_type == 'binary':
                if self.alpha_e_type == 'D':
                    alphas_e = (s_diff >= 0) * (surplus[:, None, :] >= 0)
                else:
                    alphas_e = s_diff >= 0
            # alpha^e(x,y,\tilde y)==alpha^p(y_tilde, x, y), so
            # alphas_p = np.transpose(alphas_e,(2, 0, 1))
            # g(x,\tilde y), the same for all y after integrating over \tilde y
            g_x_til_y = g[:, :, None]
            # gv(y) not gv(\tilde y), so no integration of gv.
            # (x,\tilde y,y)
            if not self.ss_fb:
                numerator = self.m(W, V) / (W * V) * alphas * np.outer(g_u, g_v) + phi * self.m(W, V) / (
                        W * V) * np.mean(
                    alphas_e * g_v[None, None, :] * g_x_til_y, axis=1)
                # alpha^e(x, y, \tilde y),
                # SS KFE denominator integrates \tilde y, so axis=2
                # alpha^e(x, y,\tilde y):=S(x,\tilde y)>S(x, y)=:alpha^p(\tilde y, x, y)
                # alphas_e[i,j,k]=(surplus[i,k]>surplus[i,j] & surplus[i,k]>0). So still
                # (x,y,\tilde y)
                # gv(\tilde y)
                denominator = eta * alphas_b + delta + phi * self.m(W, V) / (W * V) * np.mean(
                    alphas_e * g_v[None, None, :],
                    axis=2)
                g = numerator / denominator
                g = (1 - lr) * g_prev + lr * g
            else:
                mu_g = (-(eta * alphas_b + delta) * g + self.m(W, V) / (W * V) * alphas * np.outer(g_u, g_v) +
                        phi * self.m(W, V) / (W * V) * np.mean(alphas_e * g_v[None, None, :] * g_x_til_y, axis=1) -
                        phi * self.m(W, V) / (W * V) * np.mean(alphas_e * g_v[None, None, :], axis=2) * g)
                g = lr * mu_g + g_prev

            if np.isnan(g).any():
                print('g nan:', g, f'W={W},U={U}, V={V}, E={E}')
                g_nan = True
                break
            g_e, g_u, U, g_p, g_v, V, E = marginals_U_V(g)
            W = U + phi * E

            surplus_prev = surplus.copy()

            term1 = (1 - beta) * self.m(W, V) / (W * V) * np.sum(alphas * surplus * g_u[:, None], axis=0) / nx  # (ny,)
            # alpha^e(\tilde x,\tilde y,y) = S(\tilde x,y)>S(\tilde x,\tilde y)
            # g(\tilde x,\tilde y,y)<-the same in the last term y since all (\tilde x,\tilde y) have been integrated
            # S(\tilde x,\tilde y=\tilde y,y)-S(\tilde x,\tilde y,y=y). Integrating \tilde x,\tilde y
            # alphas_e[i,j,k]=(surplus[i,k]>surplus[i,j] & surplus[i,k]>0).
            term2 = (1 - beta) * phi * self.m(W, V) / (W * V) * np.sum(alphas_e * g[:, :, None] * (surplus[:, None, :] - surplus[:, :, None]),
                                                                        axis=(0, 1)) / nx / ny  # (ny,)

            # alpha^e(x,y,\tilde y) = S(x,\tilde y)>S(x,y)  & S(x,y)>0. Integrating \tilde y
            # alphas_e[i,j,k]=(surplus[i,k]>surplus[i,j] & surplus[i,k]>0). So still
            term3 = beta * phi * self.m(W, V) / (W * V) * np.sum(alphas_e * g_v[None, None, :], axis=2) / ny  # (nx,ny)

            # alpha^e(x,y,\tilde y) = S(x,\tilde y)>S(x,y)  & S(x,y)>0
            # alphas_e[i,j,k]=(surplus[i,k]>surplus[i,j] & surplus[i,k]>0). So still
            term4 = beta * phi * self.m(W, V) / (W * V) * np.sum(alphas_e * surplus[:, None, :] * g_v[None, None, :], axis=2) / ny  # (nx,ny)

            # alpha^e(x,y,\tilde y).Integrating \tilde y
            term5 = beta * self.m(W, V) / (W * V) * np.sum(alphas * surplus * g_v[None, :], axis=1) / ny  # (nx,)
            if not self.ss_fb:
                surplus = (payoffs - b - term1[None, :] - term2[None, :] + term4 - term5[:, None]) / (
                        rho + delta + eta * alphas_b + term3)
            else:
                surplus = (payoffs - b - (delta + eta * alphas_b) * surplus - term1[None, :] - term2[None, :] +
                            term4 - term3 * surplus - term5[:, None]) / rho

            surplus = (1 - lr) * surplus_prev + lr * surplus
            if np.isnan(surplus).any():
                print('S nan.term1:', term1, '\nterm2:', term2, '\nterm3:', term3, '\nterm4:', term4, '\nterm5:', term5)
                break
            if self.alpha_type == 'continuous':
                new_alphas = 1 / (1 + np.exp(-xi * surplus))
            elif self.alpha_type == 'binary':
                new_alphas = surplus >= 0
            distance = LA.norm(alphas - new_alphas, 'fro')
            if time.time() - start > 60*.75:  # stops at larger error if it takes too long
                if distance < 1e-10:
                    print(f'NonFE-DSS cannot converge at z={z} to less than 1e-13 within 0.75min, for parameters:{dict(par.items())}\nBut error is already less than 1e-10, so we will continue.')
                    break
                else:
                    if time.time() - start > 60*1.5:
                        if distance < 1e-7:
                            print(f'NonFE-DSS cannot converge at z={z} to less than 1e-10 within 1.5min, for parameters:{dict(par.items())}\nBut error is already less than 1e-7, so we will continue.')
                            break
                        else:
                            if time.time() - start > 60*3:
                                if distance <1e-5:
                                    print(f'NonFE-DSS cannot converge at z={z} to less than 1e-7 within 3min, for parameters:{dict(par.items())}\nBut error is already less than 1e-5, so we will continue.')
                                    break
                                else:
                                    raise ValueError(f'NonFE-DSS cannot converge at z={z} to less than 1e-5 within 10min, for parameters:{dict(par.items())}\nError is {distance}.')

            alphas = new_alphas
        V_v = (term1 + term2) / rho
        V_u = (term5 + b) / rho
        V_p = (1 - beta) * surplus + V_v[None, :]
        V_e = beta * surplus + V_u[:, None]
        wage = payoffs - rho * V_p - (delta + eta * alphas_b) * surplus
        print(f'E(V_v)={np.mean(V_v):.4f} given firm mass={g_f[0]:.4f},kappa={kappa},U={U * 100:.5f}%,V={V * 100:.5f}%') if printer else None
        if self.no_fe:
            print('Total unemployment U is ', U) if printer else None
            print('Total vacancy V is ', V) if printer else None

            TFP = (payoffs * g).mean() / E
            e2e = phi * self.m(W, V) / W / V * g * np.mean(alphas_e * g_v[None, None, :], axis=2)  # (nx,ny)
            e2e = e2e.mean()
            u2e = self.m(W, V) / W / V * alphas * g_u[:, None] * g_v[None, :]
            u2e = u2e.mean()
            e2u = (eta * alphas_b + delta) * g
            e2u = e2u.mean()
            if self.flow_rate:
                e2e = e2e / E
                e2u = e2u / E
                u2e = u2e / U
            if self.monthly_flow:
                e2e = e2e / 12
                e2u = e2u / 12
                u2e = u2e / 12
            self.saver.save_numpy({'g': g, 'g_f': g_f, 'g_v': g_v, 'alpha': alphas, 'surplus': surplus, 'U': U, 'V': V, 'V_u': V_u, 'V_v': V_v, 'V_e': V_e, 'V_p': V_p, 'wage': wage, 'u2e': u2e, 'e2e': e2e, 'e2u':e2u, 'TFP': TFP}, par, z, suffix)
        if save_c:
            self.saver.save_numpy({'c_par': np.mean(V_v)}, par, par.z_0, suffix)
        return alphas, alphas_e, surplus, g, np.mean(V_v), V_u, V_v, V_e, V_p, wage


    def create_g_free_entry(self, z, par=None, suffix='', printer=False):
        """Solve the fixed pt steady state with free entry."""
        nx, ny = self.nx, self.ny
        par = self.par_draw if par is None else par
        rho, kappa, beta, nu, phi, xi_e, c, eta, delta = par.rho, par.kappa, par.beta, par.nu, par.phi, par.xi_e, par.c, par.eta, par.delta
        g_w = self.gw_np

        def marginals_U_V(g, g_w, g_f=None, g_v=None):
            g_e = np.mean(g, axis=1)
            g_u = g_w - g_e
            U = np.mean(g_u)
            g_p = np.mean(g, axis=0)
            if (g_f is not None) and (g_v is None):
                g_v = g_f - g_p  # no gf(y) in Lise&Robin(2017)
            V = np.mean(g_v)
            E = np.mean(g_e)
            return g_e, g_u, U, g_p, g_v, V, E

        surplus = np.random.rand(nx, ny) * 0.9 - 0.5  # initialize S(x,y) to (-0.4,0.8)
        types_x = np.linspace(1 / nx / 2, 1 - 1 / nx / 2, nx)
        types_y = np.linspace(1 / ny / 2, 1 - 1 / ny / 2, ny)
        payoffs = z * self.calc_f(np.expand_dims(types_x, axis=1), np.expand_dims(types_y, axis=0))  # also what LR claim in their text
        dis = sys.float_info.max
        start = time.time()
        g, alphas_e, alphas_b = None, None, None  # only initialize g in create_g_no_free_entry in outside round 0. Use previous g in next iterations. Only b_r has α^e
        gf_path = os.path.join(self.path, 'gf_ss.npy')
        if os.path.exists(gf_path):
            g_f = np.load(gf_path)  # use gv under z_0 as initial guess if it exists
        else:
            g_f = (self.init_gf + np.random.uniform(-1, 1) * 0.1) * np.ones(ny)
        while dis > 1e-13:
            alphas, alphas_e, surplus, g, _, V_u, V_v, V_e, V_p, wage = self.create_g_no_free_entry(z, g_f, par=par, g=g, surplus=surplus,
                                                                                                    alphas_e=alphas_e, payoffs=payoffs)
            g_e, g_u, U, g_p, g_v, V, E = marginals_U_V(g, g_w, g_f)
            W = U + phi * E
            A = (1 - beta) * np.mean(alphas * surplus * g_u[:, None])  # scalar
            B = (1 - beta) * phi * np.mean(alphas_e * g[:, :, None] * (surplus[:, None, :] - surplus[:, :, None]), axis=(0, 1))  # (ny,),y
            B = np.mean(B)  # integrate y
            V_1 = (kappa / rho / c / W) ** (1 / nu) * W * (A + B) ** (1 / nu)
            P = np.mean(g_p, axis=0)
            g_f_1 = (V_1 + P) * np.ones(ny)
            g_f = (g_f + g_f_1) / 2
            dis = np.linalg.norm(g_f - g_f_1)

            # Timeout handling
            timeout_thresholds = [
                (60 * 0.75, 1e-10, 1e-13),
                (60 * 1.5, 1e-7, 1e-10),
                (60 * 3, 1e-5, 1e-7),
                (60 * 5, 1e-2, 1e-5)
            ]
            elapsed = time.time() - start
            for timeout, current_thresh, next_thresh in timeout_thresholds:
                if elapsed > timeout:
                    if dis < current_thresh:
                        print(f'FE-DSS cannot converge at z={z} to less than {next_thresh} within {timeout/60}min, '
                            f'for parameters:{dict(par.items())}\nBut error is already less than {current_thresh}, so we will continue.')
                        break
                    elif timeout == timeout_thresholds[-1][0]:  # Last threshold
                        raise ValueError(f'FE-DSS cannot converge at z={z} to less than {next_thresh} within {timeout/60}min, '
                                    f'for parameters:{dict(par.items())}\nError is {dis}.')
        print('g_f:', g_f) if printer else None

        # Calculate DSS statistics/moments
        TFP = (payoffs * g).mean() / E
        e2e = phi * self.m(W, V) / W / V * g * np.mean(alphas_e * g_v[None, None, :], axis=2)  # (nx,ny)
        e2e = e2e.mean()
        u2e = self.m(W, V) / W / V * alphas * g_u[:, None] * g_v[None, :]
        u2e = u2e.mean()
        alphas_b = (1 - alphas) * self.alpha_b_type 
        e2u = (eta * alphas_b + delta) * g
        e2u = e2u.mean()
        if self.flow_rate:
            e2e = e2e / E
            e2u = e2u / E
            u2e = u2e / U
        if self.monthly_flow:
            e2e = e2e / 12
            e2u = e2u / 12
            u2e = u2e / 12
        print('Total unemployment U is ', U * 100, '%') if printer else None
        print('Total vacancy V is ', V * 100, '%') if printer else None
        print('Total vacancy V1 is ', V_1 * 100, '%', '\n\n') if printer and np.abs(V_1 - V) > 1e-10 else None
        print('e2e is', e2e * 100, '%') if printer else None
        print('e2u is', e2u * 100, '%') if printer else None
        print('u2e is ', u2e * 100, '%') if printer else None

        self.saver.save_numpy({'surplus': surplus, 'g': g}, par, z, suffix)
        if suffix == '':  # only save variables below for illustration 3 cases, where suffix is empty
            self.saver.save_numpy({'U': U, 'V': V, 'u2e': u2e, 'e2e': e2e, 'e2u':e2u, 'g_f': g_f, 'g_v': g_v, 'alpha': alphas, 'V_u': V_u, 'V_v': V_v, 'V_e': V_e, 'V_p': V_p, 'wage': wage, 'TFP': TFP}, par, z, suffix)
        
        if z == par.z_0 and printer:
            y_star = np.argmax(surplus, axis=1)
            print('y*:', y_star)   


    def compute_single_steady_state(self, args):
        """Compute steady state for one (z, parameter) pair in parallel.
        
        Args:
            args: Tuple containing:
                z: State value to compute
                par: Parameter set
                suffix: Identifier for saving results
        """
        z, par, suffix = args
        
        # Handle market tightness constraint case
        if self.c_tightness:
            if z == par.z_0:
                # Compute base case that determines 'c' value
                self.create_g_no_free_entry(
                    z, 
                    np.ones(self.ny) * self.init_gf, 
                    par=par, 
                    suffix=suffix, 
                    save_c=True
                )
                return
            else:
                # Load precomputed 'c' value for other states
                c = self.tensor_loader(f'c_par{suffix}_ss.npy', to_np=True)
                par.update({'c': c.__float__()})
        
        # Compute steady state based on free entry setting
        if not self.no_fe:
            self.create_g_free_entry(z, par, suffix=suffix)
        else:
            self.create_g_no_free_entry(
                z, 
                np.ones(self.ny) * self.init_gf, 
                par=par, 
                suffix=suffix
            )

    def compute_all_steady_states(self, pars, num_processes=64, tasks=None):
        """Compute steady states for all parameter sets in parallel.
        
        Args:
            pars: List of parameter sets
            num_processes: Max parallel processes to use
            tasks: Predefined tasks if available
        """
        max_processes = multiprocessing.cpu_count()
        num_processes = min(num_processes, max_processes)
        print(f"Using {num_processes} of {max_processes} available CPUs")

        # Create computation tasks
        if tasks is None:
            tasks = []
            # Add tasks for z0 ± dz cases
            tasks.extend([(par.z_0 + par.dz, par, i) for i, par in enumerate(pars)])
            tasks.extend([(par.z_0 - par.dz, par, i) for i, par in enumerate(pars)])
            
            # Add base z0 case for tightness constraint
            if self.c_tightness:
                tasks.extend([(par.z_0, par, i) for i, par in enumerate(pars)])

        # Execute in parallel
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(self.compute_single_steady_state, tasks)
            pool.close()
            pool.join()