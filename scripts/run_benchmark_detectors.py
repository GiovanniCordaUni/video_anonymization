import argparse
from pathlib import Path

from src.pipeline.video_pipeline import load_config, build_detector_from_config
from src.utils.io_utils import open_video
from src.evaluation.ground_truth_loader import load_video_ground_truth
from src.evaluation.metrics import (
    evaluate_frame_detections,
    aggregate_metrics,
)


def benchmark_detector(config_path: str):
    # Carica config
    cfg = load_config(config_path)

    paths_cfg = cfg["paths"]
    eval_cfg = cfg["evaluation"]
    video_cfg = cfg.get("video", {})

    input_videos_dir = Path(paths_cfg["input_videos"])
    gt_root = Path(paths_cfg["ground_truth"])
    results_root = Path(paths_cfg["results"])
    results_root.mkdir(parents=True, exist_ok=True)

    iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
    score_threshold = float(eval_cfg.get("score_threshold", 0.5))
    allow_multi = bool(eval_cfg.get("allow_multiple_predictions", False))
    frame_stride = int(video_cfg.get("frame_stride", 1))

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Video di input (GT): {input_videos_dir}")
    print(f"[INFO] Ground truth root:   {gt_root}")
    print(f"[INFO] Results dir:         {results_root}")
    print(f"[INFO] IoU threshold:       {iou_threshold}")
    print(f"[INFO] Score threshold:     {score_threshold}")
    print(f"[INFO] Allow multiple pred: {allow_multi}")
    print(f"[INFO] Frame stride (eval): {frame_stride}")

    # Inizializza detector da config
    detector = build_detector_from_config(cfg)

    per_frame_stats = []

    # 3) Loop sui video nella cartella input_videos
    video_files = sorted(
    p for p in input_videos_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )

    if not video_files:
        print(f"[WARN] Nessun video trovato in {input_videos_dir}")
        return

    for video_path in video_files:
        print(f"\n[INFO] Video: {video_path.name}")

        # Apri video per ottenere dimensione frame
        cap, fps, (w, h), total_frames = open_video(str(video_path))
        print(f"      FPS: {fps:.2f}, Size: {w}x{h}, Frames: {total_frames}")

        # Carica GT per questo video (usa w,h per convertire YOLO -> pixel)
        frame_to_gt = load_video_ground_truth(
            str(video_path),
            str(gt_root),
            frame_size=(w, h),
        )

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # sottocampionamento per velocizzare
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            gt_boxes = frame_to_gt.get(frame_idx, [])

            # Predizioni del detector
            detections = detector.detect(frame)

            # filtra per confidenza
            pred_boxes = [
                (d.x1, d.y1, d.x2, d.y2)
                for d in detections
                if d.confidence >= score_threshold
            ]

            if frame_idx == 0:
                print("DEBUG w,h =", w, h)
                print("DEBUG GT sample:", gt_boxes[:3])
                print("DEBUG PRED sample:", pred_boxes[:3])

                def box_stats(boxes, name):
                    if not boxes:
                        print(f"DEBUG {name}: empty")
                        return
                    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
                    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
                    print(f"DEBUG {name}: x in [{min(xs):.3f},{max(xs):.3f}]  y in [{min(ys):.3f},{max(ys):.3f}]")

                box_stats(gt_boxes, "GT")
                box_stats(pred_boxes, "PRED")

                # sanity check su un paio di IOU
                from src.evaluation.metrics import compute_iou
                if gt_boxes and pred_boxes:
                    print("DEBUG IOU(gt0,pred0) =", compute_iou(gt_boxes[0], pred_boxes[0]))

            tp, fp, fn = evaluate_frame_detections(
                gt_boxes,
                pred_boxes,
                iou_threshold=iou_threshold,
                allow_multiple_predictions=allow_multi,
            )

            per_frame_stats.append({"tp": tp, "fp": fp, "fn": fn})

            frame_idx += 1

        cap.release()

    # Aggrega risultati
    agg = aggregate_metrics(per_frame_stats)

    print("\n========== RISULTATI GLOBALI ==========")
    print(f"TP:        {agg['tp']}")
    print(f"FP:        {agg['fp']}")
    print(f"FN:        {agg['fn']}")
    print(f"Precision: {agg['precision']:.4f}")
    print(f"Recall:    {agg['recall']:.4f}")
    print(f"F1-score:  {agg['f1']:.4f}")
    print("=======================================")

    # Salva su file
    out_file = results_root / "detector_benchmark.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("Detector benchmark results\n\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Videos dir: {input_videos_dir}\n")
        f.write(f"GT dir: {gt_root}\n\n")
        f.write(f"IoU threshold: {iou_threshold}\n")
        f.write(f"Score threshold: {score_threshold}\n")
        f.write(f"Allow multiple predictions: {allow_multi}\n\n")
        f.write(f"TP: {agg['tp']}\n")
        f.write(f"FP: {agg['fp']}\n")
        f.write(f"FN: {agg['fn']}\n\n")
        f.write(f"Precision: {agg['precision']:.6f}\n")
        f.write(f"Recall:    {agg['recall']:.6f}\n")
        f.write(f"F1-score:  {agg['f1']:.6f}\n")

    print(f"[INFO] Risultati salvati in: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark dei detector (precision/recall/F1) su dataset di test."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path al file di configurazione YAML",
    )
    args = parser.parse_args()

    benchmark_detector(args.config)