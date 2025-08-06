from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.dtype = "float32"
    config.work_dir = ""  # Specify working directory

    config.task = ConfigDict()
    config.task.name = "noisy_linear_regression"
    config.task.n_tasks = 16
    config.task.n_dims = 8
    config.task.n_points = 16
    config.task.batch_size = 8
    config.task.data_seed = 101
    config.task.task_seed = 102
    config.task.noise_seed = 103
    config.task.data_scale = 1.0
    config.task.task_scale = 1.0
    config.task.noise_scale = 0.5

    config.model = ConfigDict()
    config.model.name = "transformer"
    config.model.n_points = 16
    config.model.n_layer = 2
    config.model.n_embd = 8
    config.model.n_head = 2
    config.model.seed = 100

    config.training = ConfigDict()
    config.training.optimizer = "adam"
    config.training.lr = 1e-3
    config.training.schedule = "triangle"
    config.training.warmup_steps = 250
    config.training.total_steps = 500

    config.eval = ConfigDict()
    config.eval.n_samples = 10
    config.eval.batch_size = 8
    config.eval.data_seed = 104
    config.eval.task_seed = 105
    config.eval.noise_seed = 106
    config.eval.every = 1000

    config.wandb = ConfigDict()
    config.wandb.project = ""  # Specify wandb project

    return config
