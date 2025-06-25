hyperparameters = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 1,
    'embedding_size': 64,
    'key_query_size': 24,
    'value_size': 32,
    'num_layers': 1,
    'dropout': 0.1,
    'num_patches': 1,
    'use_pos_encoding': True,
}

sweep_config = {
    "method": "bayes", # Can be 'grid', 'random', or 'bayes'
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [64]},
        "learning_rate": {"values": [1e-3]},
        "epochs": {"values": [1, 2, 3]},
        "embedding_size": {"values": [64]},
        "key_query_size": {"values": [24]},
        "value_size": {"values": [32]},
        "num_layers": {"values": [2, 5]},
        "dropout": {"values": [0.1]},
        "num_patches": {"values": [1, 4, 16]},
        "use_pos_encoding": {"values": [True, False]},
    },
}

run_config = {
    "project": "mlx8-week-03-transformers",
    "entity": "ewanbeattie1-n-a",
    'run_type': 'sweep',  # 'sweep', 'train' or 'test
}