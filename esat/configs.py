import configparser


# ------- SIMULATOR Configuration -------- #
sim_config = configparser.ConfigParser()
sim_config['project'] = {
    "directory": "."
}
sim_config['data'] = {
    'input_path': 'synthetic_data.csv',
    'uncertainty_path': 'synthetic_uncertainty.csv'
}
sim_config['parameters'] = {
    'seed': 42,
    'factors_n': 6,                  # Number of factors in the synthetic dataset
    'features_n': 40,                # Number of features in the synthetic dataset
    'samples_n': 200,                # Number of samples in the synthetic dataset
    'outliers': True,                # Add outliers to the dataset
    'outlier_p': 0.10,               # Decimal percent of outliers in the dataset
    'outlier_mag': 2,                # Magnitude of outliers
    'contribution_max': 10,          # Maximum value of the contribution matrix (W) (Randomly sampled from a uniform distribution)
    'noise_mean_min': 0.03,          # Minimum mean of noise added to the synthetic dataset, by feature (Randomly sampled from a normal distribution)
    'noise_mean_max': 0.05,          # Maximum mean of noise added to the synthetic dataset, by feature (Randomly sampled from a normal distribution)
    'noise_scale': 0.02,             # Scale of the noise added to the synthetic dataset
    'uncertainty_mean_min': 0.04,    # Minimum mean of the uncertainty matrix, percentage of the input dataset (Randomly sampled from a normal distribution)
    'uncertainty_mean_max': 0.06,    # Maximum mean of the uncertainty matrix, percentage of the input dataset (Randomly sampled from a normal distribution)
    'uncertainty_scale': 0.01        # Scale of the uncertainty matrix
}
sim_config['contributions'] = {
    '0': '{"curve_type":"logistic","scale": 0.1,"frequency":0.5}',
    '1': '{"curve_type":"periodic","minimum": 0.1,"maximum": 0.9,"frequency": 0.5,"scale":0.1}',
    '2': '{"curve_type":"increasing","minimum": 0.1,"maximum": 0.9,"scale":0.1}',
    '3': '{"curve_type":"decreasing","minimum": 0.1,"maximum": 0.9,"scale":0.1}'
}

# ------- RUN Configuration --------- #
run_config = configparser.ConfigParser()
run_config['project'] = {
    "name": "",
    "directory": ".",
}
run_config['data'] = {
    "input_path": "",
    "uncertainty_path": "",
    "index_col": 0
}
run_config['estimator'] = {
    "samples": 250,
    "min_k": 2,
    "max_k": 12
}
run_config['parameters'] = {
    'factors': -1,
    'method': 'ls-nmf',
    'models': 20,
    'init_method': 'col_means',
    'init_norm': True,
    'seed': 42,
    'max_iter': 20000,
    'converge_delta': 0.0001,
    'converge_n': 50,
    'verbose': False,
    'optimized': True,
    'parallel': True
}

# -------- ERROR Configuration -------- #
error_config = configparser.ConfigParser()
error_config["project"] = {
    "project_config": "",
    "constrained_config": "",
    "selected_model": -1,
}
error_config["disp"] = {
    "features": '[]',
}
error_config["bs"] = {
    "bootstrap_n": 20,
    "block_size": -1,
    "threshold": 0.6,
    "seed": '-1'
}
error_config["bsdisp"] = {
    "bootstrap_output": "",
    "bootstrap_n": 20,
    "block_size": -1,
    "threshold": 0.6,
    "seed": -1,
    "threshold_dQ": 0.1,
    "max_search": 50,
    "features": '[]'
}

# -------- CONSTRAINED Configuration -------- #
constrained_config = configparser.ConfigParser()
constrained_config["project"] = {
    "project_config": ""
}
constrained_config["parameters"] = {
    "selected_model": -1,
    "softness": '1.0',
    "max_iter": -1,
    "converge_delta": 0.1,
    "converge_n": 20
}
constrained_config["constraints"] = {
    "constraint1": '{"constraint_type":"set to zero","index":[-1,-1],"target":"feature"}',
    "constraint2": '{"constraint_type":"define limits","index":[-1,-1],"target":"feature","min_value":0,"max_value":1}',
    "constraint3": '{"constraint_type":"pull up","index":[-1,-1],"target":"feature","dQ":50}',
    "constraint4": '{"constraint_type":"pull down","index":[-1,-1],"target":"feature","dQ":50}',
    "constraint5": '{"constraint_type":"set to base value","index":[-1,-1],"target":"feature"}',
    "constraint6": '{"constraint_type":"pull to value","index":[-1,-1],"target":"feature","target_value":0,"dQ":50}'
}
constrained_config["expressions"] = {
    "expression1": "(0.66*[factor:-1|feature:-1])-(4.2*[factor:-1|feature:-1])=0,250",
    "expression2": "(0.35*[factor:-1|feature:-1])-(2.0*[factor:-1|feature:-1])-(3.7*[factor:-1|feature:-1])=0,250",
    "expression3": "(3.2*[factor:-1|feature:-1])+(1.2*[factor:-1|feature:-1])+(0.1*[factor:-1|feature:-1])+(20.0*[factor:-1|feature:-1])-(10.7*[factor:-1|feature:-1])=0,250"
}
