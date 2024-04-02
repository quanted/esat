import configparser

# ------- RUN Configuration --------- #
run_config = configparser.ConfigParser()
run_config['project'] = {
    "name": "",
    "directory": "."
}
run_config['data'] = {
    "input_path": "",
    "uncertainty_path": "",
    "index_col": 0
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
