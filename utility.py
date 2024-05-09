import facer
import cv2
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(path):
    image_cv = cv2.imread(path)
    image_facer = facer.hwc2bchw(facer.read_hwc(path)).to(device=device)
    
    return image_cv, image_facer
