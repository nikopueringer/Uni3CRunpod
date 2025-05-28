# Install dependencies
pip install -r requirements.txt
# Install the main project
pip install -e .
# Data preparation
mkdir inputs
mkdir outputs
apt update
apt install -y -qq aria2
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl -d inputs/checkpoints/body_models/smpl -o SMPL_NEUTRAL.pkl
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz -d inputs/checkpoints/body_models/smplx -o SMPLX_NEUTRAL.npz
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt -d inputs/checkpoints/gvhmr -o gvhmr_siga24_release.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt -d inputs/checkpoints/hmr2 -o epoch=10-step=25000.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth -d inputs/checkpoints/vitpose -o vitpose-h-multi-coco.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt -d inputs/checkpoints/yolo -o yolov8x.pt

# Usage (with SimpleVO)
# python tools/demo/demo.py --video=docs/example_video/tennis.mp4
# python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
# The result will be saved in outputs/demo/tennis.