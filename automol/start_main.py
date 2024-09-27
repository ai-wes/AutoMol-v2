import argparse
import logging
import sys
import json
from pathlib import Path

from automol.pipeline import run_main_pipeline

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command-line arguments into the configuration dictionary."""
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    return config

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoMol-v2: Novel molecule generation and analysis pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--input_text", type=str, help="Input text describing the desired molecule function")
    parser.add_argument("--num_sequences", type=int, help="Number of molecule sequences to generate initially")
    parser.add_argument("--optimization_steps", type=int, help="Number of optimization steps to perform")
    parser.add_argument("--score_threshold", type=float, help="Minimum score threshold for accepting generated sequences")
    parser.add_argument("--device", type=str, help="Device to use for computations (cuda or cpu)")
    parser.add_argument("--skip_description_gen", action="store_true", help="Skip the description generation phase")
    return parser.parse_args()

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("main_run.log")
        ]
    )

    logger = logging.getLogger(__name__)

    args = parse_arguments()

    # Load configuration
    CONFIG_PATH = Path(args.config)
    try:
        with open(CONFIG_PATH, 'r') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Config file not found at {CONFIG_PATH}.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        sys.exit(1)

    # Update config with command-line arguments (only if provided)
    config = merge_config_with_args(config, args)
    logger.info("Configuration merged with command-line arguments.")

    # Define an emit_progress function for the standalone script
    def emit_progress(phase: str, progress: int, message: str) -> None:
        logger.info(f"[{phase}] {progress}% - {message}")

    # Run the pipeline
    run_main_pipeline(config, emit_progress)

if __name__ == "__main__":
    main()