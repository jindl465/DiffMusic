import json
import shutil
import os
from pydub import AudioSegment

# JSON 파일 경로와 복사할 대상 디렉토리 설정
json_file_path = "/mnt/storage1/Jin/MUVideo/MUVideoInstructionsEval.json"  # JSON 파일 경로
target_directory = "/mnt/storage1/Jin/MUVideo/audioset_eval"  # 복사할 디렉토리 경로
base_path = "/mnt/storage1/Jin/MUVideo/audioset"
image_directory = "/mnt/storage1/Jin/MUVideo/audioset_video_eval/first"

# 디렉토리 내 전체 파일 개수 확인
all_files = [f for f in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, f))]
print(len(all_files))

all_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
print(len(all_files))

# # 디렉토리 내 파일 목록 확인 후 mp3 파일만 삭제
# for filename in os.listdir(target_directory):
#     file_path = os.path.join(target_directory, filename)
#     if file_path.endswith(".mp3") and os.path.isfile(file_path):
#         os.remove(file_path)
#         print(f"Deleted: {file_path}")

# # JSON 파일 읽기
# with open(json_file_path, "r") as file:
#     data = json.load(file)

# for item in data:
#     # 각 항목에서 output_file 가져오기
#     file_name = item.get("output_file")
#     output_file_path = os.path.join(base_path, file_name)

#     # output_file이 존재하고, 실제 파일이 존재하는지 확인
#     if output_file_path and os.path.isfile(output_file_path):
#         # 대상 경로 설정
#         destination_path = os.path.join(target_directory, file_name.replace('.mp3', '.wav'))
#         # 파일 복사
#         # shutil.copy(output_file_path, destination_path)
#         audio = AudioSegment.from_mp3(output_file_path)
#         audio.export(destination_path, format="wav")
#         print(f"File copied to {destination_path}")
#     else:
#         print(f"Output file for {output_file_path} not found or doesn't exist.")