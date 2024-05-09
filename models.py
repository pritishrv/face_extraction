import torch
import facer
import cv2
import numpy as np


class Face_Detector():
    def __init__(self,model_path = 'retinaface/mobilenet', device="cpu"):
        try:
            self.model = facer.face_detector(model_path, device=device)
        except:
            print("Problem Loading Detector")
    def run(self,image):
        with torch.inference_mode():
            return self.model(image)

class Face_Parser():
    def __init__(self,model_path = 'farl/lapa/448', device="cpu"):
        try:
            self.model = facer.face_parser(model_path, device=device)
        except:
            print("Problem Loadin Parser")
    def run(self,image,faces):
        with torch.inference_mode():
            return self.model(image, faces)