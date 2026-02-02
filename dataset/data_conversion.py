import imageio
from PIL import Image
import torch
import json
import os
import cv2

import os
import json
import cv2

def save_video_thumbnails(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    success, first_frame = cap.read()
    if not success or first_frame is None:
        print(f"Error extracting first frame from: {video_path}")
        cap.release()
        return
    
    middle_frame = None
    if total_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        success, middle_frame = cap.read()
        if not success or middle_frame is None:
            print(f"Error extracting middle frame from: {video_path}")
    
    cap.release()
    
    file_name = os.path.splitext(os.path.basename(video_path))[0] + ".jpg"
    first_thumbnail_path = os.path.join(video_dir_path, "first", file_name)
    middle_thumbnail_path = os.path.join(video_dir_path, "middle", file_name)
    
    os.makedirs(os.path.dirname(first_thumbnail_path), exist_ok=True)
    os.makedirs(os.path.dirname(middle_thumbnail_path), exist_ok=True)
    
    if first_frame is not None:
        cv2.imwrite(first_thumbnail_path, first_frame)
    if middle_frame is not None:
        cv2.imwrite(middle_thumbnail_path, middle_frame)
    
    print(f"Thumbnails saved: {first_thumbnail_path}, {middle_thumbnail_path}")

json_file = "/mnt/storage1/Jin/MUVideo/MUVideoInstructionsEval.json"
video_dir_path = "/mnt/storage1/Jin/MUVideo/audioset_video_eval/"

with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
if not isinstance(data, list):
    print("Error: JSON 파일은 리스트 형식이어야 합니다.")


for item in data:
    input_filepath = os.path.join(video_dir_path, item.get("input_file"))
    
    # 비디오의 썸네일 저장
    if input_filepath and os.path.exists(input_filepath):
        save_video_thumbnails(input_filepath)
