import jax
import hydra
from omegaconf import DictConfig
from absl import logging
from ml_collections import ConfigDict

import icl.utils as u
from icl.train import train

logging.set_verbosity(logging.INFO)


def omega_to_ml_collections(cfg: DictConfig) -> ConfigDict:
    """Convert OmegaConf DictConfig to ml_collections ConfigDict."""
    def _convert(obj):
        if isinstance(obj, DictConfig):
            config = ConfigDict()
            for key, value in obj.items():
                config[key] = _convert(value)
            return config
        else:
            return obj
    
    return _convert(cfg)


@hydra.main(version_base=None, config_path="icl/configs", config_name="example")
def main(cfg: DictConfig) -> None:
    logging.info(f"Process: {jax.process_index() } / {jax.process_count()}")
    logging.info("Local Devices:\n" + "\n".join([str(x) for x in jax.local_devices()]) + "\n")

    config = omega_to_ml_collections(cfg)
    config = u.filter_config(config)
    train(config)


if __name__ == "__main__":
    main()
