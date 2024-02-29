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
    "index_column": 0
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
    'converge_n': 100,
    'verbose': True,
    'optimized': True,
    'parallel': True
}

# -------- ERROR Configuration ---------#
error_config = configparser.ConfigParser()
error_config["disp"] = {
    "selected_model": '-1',
    "features": '[]',
}
error_config["bs"] = {
    "selected_model": '-1',
    "bootstrap_n": '20',
    "block_size": '-1',
    "threshold": '0.6',
    "seed": '-1'
}
error_config["bsdisp"] = {
    "boostrap_output": "",
    "selected_model": '-1',
    "block_size": '-1',
    "threshold": '0.6',
    "seed": "-1",
    "threshold_dQ": '0.1',
    "max_search": '50',
    "features": '[]'
}
