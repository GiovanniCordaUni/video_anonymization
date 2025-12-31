import os
import argparse
from pathlib import Path

from src.pipeline.video_pipeline import (
    run_dataset_from_config,
    run_single_as_dataset_from_config,
    load_config,  # <-- deve esistere nel tuo progetto
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

    PROJECT_ROOT = Path(__file__).resolve().parent

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config/config.yaml"),
        help="Path al file di configurazione YAML (relativo al progetto o assoluto)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,  # <-- non required: se None usa il config
        help="Path di input (file singolo o cartella di video). Se omesso usa il config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,  # <-- non required: se None usa il config
        help="Cartella root output. Se omesso usa il config.",
    )

    args = parser.parse_args()

    # risolvi config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovato: {config_path}")

    # carica config
    cfg = load_config(str(config_path))
    paths_cfg = cfg.get("paths", {})

    # fallback: CLI > config
    input_str = args.input if args.input is not None else paths_cfg.get("input_videos")
    output_str = args.output if args.output is not None else paths_cfg.get("output_videos")

    if not input_str:
        raise ValueError("Manca input: passa --input oppure imposta paths.input_videos nel config.yaml")
    if not output_str:
        raise ValueError("Manca output: passa --output oppure imposta paths.output_videos nel config.yaml")

    input_path = Path(input_str)
    output_root = Path(output_str)

    # se relativi: rendili relativi al progetto
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    # esegui
    if input_path.is_dir():
        run_dataset_from_config(
            config_path=str(config_path),
            input_dir=str(input_path),
            output_root=str(output_root),
        )
    else:
        run_single_as_dataset_from_config(
            config_path=str(config_path),
            input_video_path=str(input_path),
            output_root=str(output_root),
        )
