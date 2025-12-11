import os
import argparse

from src.pipeline.video_pipeline import (
    run_dataset_from_config,
    run_single_as_dataset_from_config,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Video anonymization pipeline\n\n"
            "- Se --input è un file crea:\n"
            "  output_root/soggettoNNN/NNN_exercise_session_anon.ext\n"
            "- Se --input è una cartella: elabora tutti i video nella cartella\n"
            "  e salva in struttura soggettoNNN dentro --output.\n\n"
            "NOTA: in entrambi i casi, --output è la CARTELLA ROOT dove creare\n"
            "      le sottocartelle soggettoNNN/."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione YAML",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path di input (file singolo o cartella di video)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Cartella root dove creare soggettoNNN/...",
    )

    args = parser.parse_args()

    if os.path.isdir(args.input):
        # modalità DATASET (cartella intera)
        run_dataset_from_config(
            config_path=args.config,
            input_dir=args.input,
            output_root=args.output,
        )
    else:
        # modalità file singolo con struttura soggettoNNN
        run_single_as_dataset_from_config(
            config_path=args.config,
            input_video_path=args.input,
            output_root=args.output,
        )
