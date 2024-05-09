import streamlit as st
import torch
from models import Face_Detector , Face_Parser
from utility import load_image
import cv2
import numpy as np


st.title("Face Extraction")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
temp_file = "temp_images/temp.jpg"
result_image_path = "temp_images/temp2.png"



img_file_buffer = st.file_uploader("Upload your Image here - ", type=['jpg', 'jpeg', 'png'])

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    cv2.imwrite(temp_file,image)

    image_cv, image_facer = load_image(temp_file)


    detector = Face_Detector(model_path = 'retinaface/mobilenet', device=device)

    faces = detector.run(image_facer)
    # print(len(faces['rects']))
    if len(faces['rects'])==1:
        parser = Face_Parser(model_path = 'farl/lapa/448', device=device)

        faces = parser.run(image_facer,faces)

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
        vis_img = vis_seg_probs.sum(0, keepdim=True)
        vis_img_npy = vis_img.permute(1,2,0).detach().cpu().numpy()
        vis_img_npy[vis_img_npy>0]=255
        transparent_mask = np.concatenate((image_cv,np.zeros((image_cv.shape[0],image_cv.shape[1],1))),axis=2)

        transparent_mask[:, :, 3] = vis_img_npy[:,:,0]
        cv2.imwrite(result_image_path,transparent_mask)
        st.image(result_image_path, caption='Transparent Image', use_column_width=True)
    else:
        st.write("Please Use Image with One Person")





    

    