
import pandas as pd
import json

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("scikit-learn 라이브러리가 설치되어 있지 않습니다.")
    print("터미널에 'pip install scikit-learn'을 입력하여 설치해주세요.")
    exit()

# 1. 약어 처리 및 지역명 목록 생성
sido_abbr_map = {
    '서울특별시': '서울',
    '부산광역시': '부산',
    '대구광역시': '대구',
    '인천광역시': '인천',
    '광주광역시': '광주',
    '대전광역시': '대전',
    '울산광역시': '울산',
    '세종특별자치시': '세종',
    '경기도': '경기',
    '강원특별자치도': '강원',
    '충청북도': '충북',
    '충청남도': '충남',
    '전북특별자치도': '전북',
    '전라남도': '전남',
    '경상북도': '경북',
    '경상남도': '경남',
    '제주특별자치도': '제주'
}

def get_all_regions(df_regions):
    all_regions = set()
    for _, row in df_regions.iterrows():
        sido = row.get('시도명')
        gugun = row.get('구명') # '구명' 또는 '시군구명' 등 파일에 맞게 수정
        dong = row.get('동명')

        # 원본 이름 추가
        names = {sido, gugun, dong}
        
        # 약어 생성
        sido_abbr = sido_abbr_map.get(sido)
        gugun_abbr = gugun[:-1] if gugun and gugun.endswith(('구', '군', '시')) else None
        dong_abbr = dong[:-1] if dong and dong.endswith(('동', '읍', '면', '리')) else None
        
        abbrs = {sido_abbr, gugun_abbr, dong_abbr}
        
        # 원본과 약어의 모든 유효한 조합 추가
        valid_names = {name for name in (names | abbrs) if name}
        all_regions.update(valid_names)

        # 2개 단위 조합 (예: 서울 종로구, 서울 종로)
        if sido and gugun:
            all_regions.add(f"{sido} {gugun}")
            if sido_abbr: all_regions.add(f"{sido_abbr} {gugun}")
            if gugun_abbr: all_regions.add(f"{sido} {gugun_abbr}")
            if sido_abbr and gugun_abbr: all_regions.add(f"{sido_abbr} {gugun_abbr}")

    return sorted(list(all_regions), key=len, reverse=True)

def create_bio_tags(message, regions):
    tokens = message.split()
    tags = ['O'] * len(tokens)
    
    for region in regions:
        if region in message:
            region_tokens = region.split()
            for i in range(len(tokens) - len(region_tokens) + 1):
                if tokens[i:i+len(region_tokens)] == region_tokens:
                    if all(tags[j] == 'O' for j in range(i, i + len(region_tokens))):
                        tags[i] = 'B-LOCATION'
                        for j in range(1, len(region_tokens)):
                            tags[i+j] = 'I-LOCATION'
    
    return {"tokens": tokens, "tags": tags}

def save_data_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # 데이터 로드
    try:
        df_messages = pd.read_csv('disaster_message.csv')
        df_regions = pd.read_csv('data/변환된_법정동.csv')
    except FileNotFoundError as e:
        print(f"오류: {e.filename} 파일을 찾을 수 없습니다.")
        return

    # 지역명 및 약어 목록 생성
    print("지역명 및 약어 목록 생성 중...")
    sorted_regions = get_all_regions(df_regions)
    
    # 전체 메시지에 대해 BIO 태깅 수행
    print("메시지 태깅 작업 수행 중...")
    messages = df_messages['message_content'].dropna().astype(str)
    tagged_results = [create_bio_tags(msg, sorted_regions) for msg in messages]

    # 학습 데이터와 테스트 데이터로 분리 (80:20)
    if not tagged_results:
        print("태깅된 결과가 없어 데이터를 분리할 수 없습니다.")
        return
        
    train_data, test_data = train_test_split(tagged_results, test_size=0.2, random_state=42)

    # 파일로 저장
    print("데이터 분리 및 JSON 파일로 저장 중...")
    save_data_to_json(train_data, 'train.json')
    save_data_to_json(test_data, 'test.json')

    print("--- 모든 작업 완료 ---")
    print(f"총 {len(tagged_results)}개의 메시지 처리 완료.")
    print(f"학습 데이터 {len(train_data)}개 -> train.json")
    print(f"테스트 데이터 {len(test_data)}개 -> test.json")

if __name__ == '__main__':
    main()
