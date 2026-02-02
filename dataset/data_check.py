import os
import json

video_dir_path = "/mnt/storage1/Jin/MUVideo/audioset_video_eval/"
audio_dir_path = "/mnt/storage1/Jin/MUVideo/audioset_eval/"

def check_and_delete(json_file):
    # JSON 파일 읽기
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print("Error: JSON 파일은 리스트 형식이어야 합니다.")
        return
    
    new_data = []
    for item in data:
        input_filepath = os.path.join(video_dir_path, item.get("input_file"))
        output_filepath = os.path.join(audio_dir_path, item.get("output_file").replace(".mp3", ".wav"))
        
        # 파일 존재 여부 확인
        if not os.path.exists(input_filepath) or not os.path.exists(output_filepath):
            to_delete = {input_filepath, output_filepath}
            
            # 파일 삭제 실행
            for file in to_delete:
                if file and os.path.exists(file):
                    os.remove(file)
                    print(f"Deleted: {file}")
        else:
            new_data.append(item)
    
    # JSON 파일 업데이트
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)
    
    print("파일 정리 완료.")

# 사용 예시
json_file = "/mnt/storage1/Jin/MUVideo/MUVideoInstructionsEval.json"
check_and_delete(json_file)