#!/usr/bin/env python3
"""
Main training script for GPT2 on linear regression data.
"""

import hydra
from omegaconf import DictConfig
import jax
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(config)
    
    # Log JAX devices
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"JAX local device count: {jax.local_device_count()}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(config)
    
    # Start training
    logger.info("Starting training process...")
    final_state = trainer.train()
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_metrics = trainer.evaluate()
    
    logger.info("Final evaluation results:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()