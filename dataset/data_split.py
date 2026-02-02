import json
from sklearn.model_selection import train_test_split

# JSON 데이터 파일 경로pip
json_path = '/mnt/storage1/Jin/MUVideo/MUVideoInstructions.json'
train_output_path = '/mnt/storage1/Jin/MUVideo/MUImageInstructionsTrain.json'
eval_output_path = '/mnt/storage1/Jin/MUVideo/MUImageInstructionsEval.json'

# 데이터셋 로드
with open(json_path, 'r') as file:
    data = json.load(file)

# 데이터셋을 80:20 비율로 train과 eval로 분리
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

# 분리된 데이터셋을 각각의 JSON 파일로 저장
with open(train_output_path, 'w') as train_file:
    json.dump(train_data, train_file, indent=2)

with open(eval_output_path, 'w') as eval_file:
    json.dump(eval_data, eval_file, indent=2)

print("JSON files for train and eval splits created successfully.")

# import pandas as pd
# from sklearn.model_selection import train_test_split

# # CSV 파일 로드
# file_path = "/mnt/storage1/Jin/MUVideo/MUVideoInstructions.json"  # 여기에 CSV 파일 경로 입력
# df = pd.read_csv(file_path)

# # 90% Train, 10% Test로 분할
# train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

# # 결과 저장
# train_file = "/mnt/storage1/Jin/MUVideo/MUVideoInstructionsTrain.json"
# test_file = "/mnt/storage1/Jin/MUVideo/MUVideoInstructionsEval.json"

# train_df.to_csv(train_file, index=False)
# test_df.to_csv(test_file, index=False)

# print(f"Train 데이터 저장 완료: {train_file}")
# print(f"Test 데이터 저장 완료: {test_file}")

