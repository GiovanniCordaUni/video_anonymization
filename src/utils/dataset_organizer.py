from pathlib import Path
import shutil


def _parse_name(filename: Path):
    """
    Converte nomi tipo:
        4sst-Andrea.mp4
        4sst-Anna1.mp4
        4sst-Ragazzo-blu2-3.mp4

    in: (exercise, subject, session)
    """
    stem = filename.stem          # stem: nome del file senza estensione es. "4sst-Andrea1"
    exercise, rest = stem.split("-", 1)  # "4sst", "Andrea1"

    # Se c'è un "_" lo uso come separatore soggetto/sessione
    if "_" in rest:
        subject, session = rest.split("_", 1)  # "Andrea_cut" -> "Andrea", "cut"
    else:
        # separo lettere e numeri finali
        letters = []
        digits = []
        for ch in rest:
            if ch.isdigit():
                digits.append(ch)
            else:
                letters.append(ch)
        subject = "".join(letters).rstrip("-")
        session = "".join(digits) or "1"   # se non ci sono numeri -> default "1"

    subject = subject.strip()
    session = session.strip()

    return exercise, subject, session


def organize_videos(input_dir: str, output_dir: str):
    """
    Riorganizza i .mp4 in soggettoNNN/ NNN_exercise_session.mp4:

        output_dir/
            soggetto001/001_4sst_1.mp4
            soggetto002/002_4sst_2.mp4
            ...

    L'ID soggetto (001, 002, ...) è assegnato nell'ordine in cui i nomi compaiono.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    subject_to_id = {}
    next_id = 1

    for src in sorted(input_path.glob("*.mp4")):
        exercise, subject, session = _parse_name(src)

        if subject not in subject_to_id:
            subject_to_id[subject] = next_id
            next_id += 1

        sid = subject_to_id[subject]
        sid_str = f"{sid:03d}"

        subject_dir = output_path / f"soggetto{sid_str}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        dst_name = f"{sid_str}_{exercise}_{session}{src.suffix}"
        dst = subject_dir / dst_name

        print(f"{src.name}  ->  {subject_dir.name}/{dst_name}")
        shutil.copy2(src, dst)   

    print("\nMappa soggetti:")
    for name, sid in sorted(subject_to_id.items(), key=lambda x: x[1]):
        print(f"  soggetto{sid:03d}  <-  {name}")