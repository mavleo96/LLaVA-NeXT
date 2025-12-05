Training Notes:
Epochs: 1
Batch Size: 32 -> (device 2 * batch size 4 * accum steps 4)
Lora: R=32, alpha=64, dropout=0.05, bias=none

Image Aspect Ratio: nobase
Image Grid Pinpoints: (1x1),...,(6x6)
Image Merge Type: spatial_unpad

Learning Rate: 1e-5
Weight Decay: 0.0
Warmup Ratio: 0.03
LR Scheduler Type: cosine
Logging Steps: 100
Save Steps: 1000
Save Total Limit: 20

Datasets: Subset of LLaVA-OneVision-Data
Cauldron:
aokvqa: 16534 (knowledge based vqa)
chartqa: 18260 (chart based vqa)
clevr: 69995 (solid geometric objects based position vqa)
tqa/iconqa(clash): 27302 (icons)
raven: 41995 (iq questions)
visual7w: 14361 (general image captioning)

Vision Flan: 186060 (vision based multiple choice questions)
Image Textualization: 99573 (image textualization)

Total: 188447 (cauldron) + 186060 (vision flan) + 99573 (captioning) = 474080

GPU Details:
(llava2) root@C.28511690:/workspace/LLaVA-NeXT$ nvidia-smi
Fri Dec  5 05:19:05 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0 Off |                  Off |
| 53%   61C    P0            207W /  450W |    8471MiB /  49140MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        On  |   00000000:41:00.0 Off |                  Off |
| 43%   62C    P0            207W /  430W |   25215MiB /  49140MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+