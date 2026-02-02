import json
import re
from collections import Counter

# --- 1. 장르 온톨로지 및 키워드 정의 ---
# 'Parent_Genre': ['keyword1', 'keyword2', ...]
# (중요) 'rock and roll'처럼 긴 키워드를 'rock'보다 먼저 배치해야 합니다.
GENRE_ONTOLOGY = {
    'Rock': [
        'rock and roll', 'blues-rock', 'hard rock', 'folk rock', 'pop rock', 
        'metal', 'heavy metal', 'rock song', 'rock', 'punk', 'grungy'
    ],
    'Blues': ['blues', 'bluesy', 'country blues'],
    'Jazz': ['jazz', 'jazz fusion', 'swing', 'big band', 'jazzy', 'bossa nova'],
    'Classical': [
        'classical', 'orchestral', 'symphony', 'baroque', 'classical piece',
        'strings', 'string section', 'violin', 'cello', 'viola', 'harp'
    ],
    'Electronic': [
        'electronic', 'techno', 'trance', 'ambient', 'new age', 'synth',
        'synthesizer', 'edm', 'house', 'industrial', 'electro', 
        'electronic dance music'
    ],
    'Folk/Country': [
        'folk', 'country', 'bluegrass', 'folk song', 'acoustic', 'banjo', 
        'mandolin', 'ukulele', 'fiddle'
    ],
    'Pop': ['pop', 'pop song', 'synth-pop', 'k-pop'],
    'Funk/Soul': ['funk', 'funky', 'soul', 'soulful', 'groovy', 'r&b', 'disco'],
    'Hip Hop': ['hip hop', 'rap'],
    'Latin': ['latin', 'salsa', 'latin dance'],
    'World': [
        'indian classical', 'indian', 'sitar', 'tabla', 'didgeridoo', 
        'bagpipes', 'traditional folk', 'aboriginal', 'arabic'
    ],
    'Other': ['lullaby', 'jig', 'experimental', 'fusion', 'ambient', 'drone']
}

def build_keyword_map():
    """
    온톨로지에서 { 'keyword': 'Parent_Genre' } 맵과
    정규식 검색을 위한 정렬된 키워드 리스트를 생성합니다.
    """
    parent_map = {}
    all_keywords = []
    
    for parent, children in GENRE_ONTOLOGY.items():
        for child in children:
            parent_map[child] = parent
            all_keywords.append(child)
            
    # 긴 키워드가 먼저 매치되도록 정렬 (예: "folk rock" > "rock")
    sorted_keywords = sorted(all_keywords, key=len, reverse=True)
    return parent_map, sorted_keywords

def get_gt_genre(text: str, keywords_list: list, parent_map: dict) -> str:
    """
    텍스트에서 첫 번째로 발견되는 장르 키워드의 상위 장르를 반환합니다.
    """
    text_low = text.lower()
    for keyword in keywords_list:
        # \b: 단어 경계를 확인하여 'rock'이 'rocket'에 매치되는 것을 방지
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_low):
            return parent_map[keyword]
    return None # 매치되는 장르 없음

def process_muimage_json(json_file: str, keywords_list: list, parent_map: dict):
    """
    JSON 파일 전체를 파싱하여 GT 장르 레이블을 추출합니다.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON 파일 '{json_file}'을 찾을 수 없습니다.")
        print("스크립트와 JSON 파일이 같은 폴더에 있는지 확인하세요.")
        return None
    except json.JSONDecodeError:
        print(f"Error: '{json_file}'이 올바른 JSON 형식이 아닙니다.")
        return None

    gt_labels = []
    
    for item in data:
        input_file = item.get('input_file', 'unknown')
        conversation = item.get('conversation', [])
        
        if len(conversation) < 2:
            continue
            
        # 1. Human 요청 텍스트
        human_prompt = conversation[0].get('value', '')
        # 2. GPT (GT) 설명 텍스트
        gpt_caption = conversation[1].get('caption', '')
        
        # Human 요청에서 먼저 장르 탐색
        gt_genre = get_gt_genre(human_prompt, keywords_list, parent_map)
        
        # 못찾았으면 GPT 설명에서 탐색
        if not gt_genre:
            gt_genre = get_gt_genre(gpt_caption, keywords_list, parent_map)
        
        # 그래도 못찾았으면 'Other'
        if not gt_genre:
            gt_genre = 'Other'
            
        gt_labels.append({
            'input_file': input_file,
            'gt_genre_label': gt_genre,
            'human_prompt': human_prompt,
            'gpt_caption': gpt_caption
        })
        
    return gt_labels

# --- 4. 메인 스크립트 실행 ---
if __name__ == "__main__":
    JSON_FILE_PATH = '/mnt/storage1/Jin/MUImage/MUImageInstructionsEval.json'
    
    # 온톨로지 및 키워드 리스트 빌드
    parent_map, sorted_keywords = build_keyword_map()
    
    # 데이터 처리
    ground_truth_data = process_muimage_json(JSON_FILE_PATH, sorted_keywords, parent_map)
    
    if ground_truth_data:
        print(f"✅ 성공: {len(ground_truth_data)}개의 항목에서 GT 장르 레이블 추출 완료.\n")
        
        # 샘플 15개 출력
        print("--- 추출된 GT 레이블 샘플 (상위 15개) ---")
        for item in ground_truth_data[:15]:
            print(f"  [ {item['input_file']} ] -> {item['gt_genre_label']}")
            
        # 전체 장르 분포 요약
        genre_counts = Counter([item['gt_genre_label'] for item in ground_truth_data])
        print("\n--- 전체 장르 분포 요약 ---")
        for genre, count in genre_counts.most_common():
            print(f"  {genre:<15} : {count}개")
            
        # 결과를 파일로 저장
        output_filename = 'ground_truth_genres.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, indent=2)
            
        print(f"\n✅ 전체 결과가 '{output_filename}' 파일로 저장되었습니다.")
        print("이제 이 GT 레이블을 '예측 레이블'과 비교하여 정확도를 계산할 수 있습니다.")