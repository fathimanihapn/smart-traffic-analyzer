# Test installed libraries

# Test numpy
import numpy as np
print("✅ Numpy working:", np.array([1, 2, 3]))

# Test OpenCV
import cv2
print("✅ OpenCV version:", cv2.__version__)

# Test PyTorch
import torch
print("✅ Torch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

# Test Ultralytics YOLO
from ultralytics import YOLO
print("✅ YOLO loaded successfully")

# Test Matplotlib
import matplotlib
print("✅ Matplotlib version:", matplotlib.__version__)

print("\n🎉 All libraries imported successfully!")
