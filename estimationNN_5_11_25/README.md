run.py is the main script that runs both the surplus NN training (train_nn.py), as well as the calibration that runs the estimation (calibrator.py).

Note: still not sure how c_tightness fits into this, will ask Runhuan about this.

The flow of calculations carried out by run.py are as follows:

Using train_nn.py to solve the augmented Surplus Master equation...
- Deterministic steady state (DSS) Solutions:
    1) Solve for the deterministic steady state (DSS) for different 'M_par_ss' number of parameter sets,
    which are sampled at random according to the parameters bounds in the config file. For each parameter set, the DSS is solved three times depending on the aggregate state, particularly at z_low = par.z_0 - par.dz and z_high = par.z_0 + par.dz. Save all the DSS densities and surplus at each of these aggregate states in the consolidated files 'g_highs.pt', 'g_lows.pt', 's_highs.pt', 's_lows.pt'.
    
    2) Use the parameter sets and steady state solution files 'g_highs.pt', 'g_lows.pt', 's_highs.pt', 's_lows.pt' acquired in the previous step to train a Steady_State_Surrogate NN that maps parameter sets to a vector (g(x,y,z), S(x,y,z)).  This vector has length nx * ny * nz * 2 that concatenates flattened matching density and the surplus function at the DSS.


- Pretrain/warm start the Surplus NN: 
    Note: Surplus_NN maps vectors (x, y, z, g, pars) of length 3 + n_x * n_y + n_par to a scalar containing the surplus value at that input vector.

    3) Generate random input vectors (x, y, z, g, pars) and use Steady_State_Surrogate NN to obtain the DSS S(x, y, z) at the parameter values. This will be used as an 'initial guess' for the Surplus_NN.

    4) Train Surplus_NN to learn a mapping to DSS surplus values.


- Train the Surplus NN to minimize the master equation.     Within the main training loop of the surplus function repeat the following steps each epoch:

    Sample training points (see sample() and g_sampler())
    - 5) Draw a set of parameters at random, again according to par_range in the config file.
    - 6) Draw worker and firm types x, y according to a discrete uniform multinomial distribution
    - 7) Draw aggregate state z uniformly at random (0 for low, 1 for high)
    - 8) Sample worker-firm match density values (g) that are perturbed random combinations of the DSS g corresponding to the parameters drawn. Each sample is drawn from a truncated normal distribution centered between g_low and g_high, where g_low and g_high are taken using the Steady_State_Surrogate NN.    
        More specifically:
            g ~ TruncatedNormal(μ, σ²)
            where:
                μ = (g_high + g_low)/2
                σ = (g_high - g_low)/std_size
            with truncation at:
                lower bound: max(0, μ - upper_std*σ)
                upper bound: μ + upper_std*σ
    
    - 9) Evaluate the Master equation loss on the sampled training points (see S_pde_oper())

    - 10) Take an optimization/backprop step according to the calculated loss.

    - 11) Break if the surplus loss threshold is reached.

    

Once the Surplus_NN has been trained, we then calibrate the model with calibrator.py...

Note: the ergodic aggregate moments are: ergodic unemployment rate, vacancy rate, employment-
to-employment transition rate, unemployment-to-employment transition rate, and
employment-to-unemployment transition rate.

- Train a new calibrator_NN to learn a mapping from parameter space to ergodic aggregate moments:
    1) First crete a train and validation dataset using the generate_and_save_chunks(). This process involves drawing many parameter sets,
    which are sampled at random according to the parameters bounds in the config file, using the previously trained surplus NN and simulate_class to find the ergodic moments. This creates the simulated dataset.

    2) Using the simulated training and validation datasets, minimize the loss of the calibrator_NN (sum of MSE losses for all moments) until the loss is below the threshold or the maximum number of iterations is reached.

- Find the optimal parameters using the newly trained model.
    1) This calls the function find_optimal_par() which makes use of the moment_distance() helper function.The moment_distance method calculates the weighted squared distance between calibrator_NN model-predicted moments and predefined target moments.vThe find_optimal_par method uses gradient-based optimization (AdamW) to adjust input parameters for a given model, minimizing the moment_distance between the model's output moments and target moments, while respecting parameter bounds. Together, these methods enable parameter estimation by fitting a model's moments to target moments by optimizing the parameters to achieve the best fit.

    2) After finding the 'optimal' parameters, run a final simulation with these parameters and surplus_NN to calculate the simulated ergodic moments and solution (i.e. ergodic density g, and the surplus, alpha functions at this g).
