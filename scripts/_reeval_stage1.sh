set -e
PY=.venv/Scripts/python.exe
rm -f results/stage1/all_runs.csv
for det in yolov9m yolov10m yolo11m yolo12m yolo26m rtdetr_l; do
  for seed in 0 1 2 3 4; do
    echo "===== EVAL $det seed$seed ====="
    $PY scripts/01_eval_detector.py \
      --weights runs/stage1_detectors/${det}_seed${seed}/weights/best.pt \
      --config configs/detectors/${det}.yaml
  done
done
echo "===== AGGREGATE ====="
$PY -c "import sys; sys.path.insert(0,'scripts'); from run_benchmark import aggregate_runs; from pathlib import Path; aggregate_runs(Path('results/stage1/all_runs.csv'), Path('results/stage1/aggregated.csv'))"
echo "===== REEVAL DONE ====="
