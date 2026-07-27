#!/bin/bash
export LD_LIBRARY_PATH=/home/aiadmin02/id_fin_detection/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/aiadmin02/id_fin_detection/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/aiadmin02/id_fin_detection/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/aiadmin02/id_fin_detection/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
.venv/bin/streamlit run app.py "$@"
