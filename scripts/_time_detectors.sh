set -e
PY=.venv/Scripts/python.exe
for det in yolov9m yolov10m yolo11m yolo12m yolo26m rtdetr_l; do
  $PY scripts/_time_detectors.py --detector $det
done
echo "ALL TIMING DONE"
