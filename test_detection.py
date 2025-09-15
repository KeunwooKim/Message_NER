import pandas as pd
import re
from thefuzz import process

print("--- 지역명 감지 기능 테스트를 시작합니다. ---")

# --- 1. 기준 데이터 준비 ---
try:
    df_location = pd.read_csv('data/변환된_법정동_계층_코드포함.csv')
    print("✓ 기준 데이터(변환된_법정동_계층_코드포함.csv) 로드 완료")
    df_disaster = pd.read_csv('disaster_message.csv')
    print("✓ 테스트용 재난 문자 데이터(disaster_message.csv) 로드 완료")
except FileNotFoundError as e:
    print(f"오류: 필수 CSV 파일이 없습니다. {e}")
    exit()

# 데이터 전처리
for col in ['시도명', '시군구명', '읍면동명', '리명']:
    df_location[col] = df_location[col].fillna('')

df_sigungu_level = df_location[df_location['시군구명'].ne('') & df_location['읍면동명'].eq('')].copy()
unique_sido_names = df_sigungu_level['시도명'].unique().tolist()
sigungu_choices_by_sido = {sido: df_sigungu_level[df_sigungu_level['시도명'] == sido]['시군구명'].unique().tolist() for sido in unique_sido_names}
print("✓ 검색 후보 목록 생성 완료")


# --- 2. 테스트할 함수 --- 
def find_location_code_hierarchical(message_text):
    # 1. 불필요한 단어 제거 및 검색할 단어 목록 생성
    ignore_words = ['오늘', '오전', '오후', '[Web발신]', '[중대본]', '[행정안전부]', '현재']
    clean_message = re.sub(r'\d{2,4}[년\-/.]\d{1,2}[월\-/.]\d{1,2}일?', '', message_text)
    clean_message = re.sub(r'\d{1,2}:\d{1,2}', '', clean_message)
    
    parts = clean_message.split()
    search_parts = [p for p in parts if p not in ignore_words and '(' not in p and ')' not in p][:7]

    best_sido = None
    best_sigungu = None
    
    # 2. 시도/시군구 후보 찾기
    for i, part in enumerate(search_parts):
        sido_match = process.extractOne(part, unique_sido_names, score_cutoff=85)
        if sido_match:
            if i + 1 < len(search_parts):
                sigungu_choices = sigungu_choices_by_sido.get(sido_match[0], [])
                if i + 2 < len(search_parts):
                    sigungu_candidate = f"{search_parts[i+1]} {search_parts[i+2]}"
                    sigungu_match = process.extractOne(sigungu_candidate, sigungu_choices, score_cutoff=85)
                    if sigungu_match:
                        best_sido = sido_match[0]
                        best_sigungu = sigungu_match[0]
                        break
                
                sigungu_match = process.extractOne(search_parts[i+1], sigungu_choices, score_cutoff=85)
                if sigungu_match:
                    best_sido = sido_match[0]
                    best_sigungu = sigungu_match[0]
                    break
    
    if not (best_sido and best_sigungu):
        return None, None, None, None, None

    best_match_row = None
    base_result = df_sigungu_level[(df_sigungu_level['시도명'] == best_sido) & (df_sigungu_level['시군구명'] == best_sigungu)]
    if not base_result.empty:
        best_match_row = base_result.iloc[0]

    eupmyeondong_df = df_location[(df_location['시도명'] == best_sido) & (df_location['시군구명'] == best_sigungu) & (df_location['읍면동명'] != '')]
    unique_eupmyeondongs = eupmyeondong_df['읍면동명'].unique()
    
    found_eupmyeondong = None
    for name in unique_eupmyeondongs:
        if name in message_text:
            found_eupmyeondong = name
            eup_result = eupmyeondong_df[(eupmyeondong_df['읍면동명'] == name) & (eupmyeondong_df['리명'] == '')]
            if not eup_result.empty:
                best_match_row = eup_result.iloc[0]
            break

    if found_eupmyeondong:
        ri_df = eupmyeondong_df[(eupmyeondong_df['읍면동명'] == found_eupmyeondong) & (eupmyeondong_df['리명'] != '')]
        unique_ris = ri_df['리명'].unique()
        
        for name in unique_ris:
            if name in message_text:
                ri_result = ri_df[ri_df['리명'] == name]
                if not ri_result.empty:
                    best_match_row = ri_result.iloc[0]
                break

    if best_match_row is not None:
        return best_match_row['법정동코드'], best_match_row['시도명'], best_match_row['시군구명'], best_match_row['읍면동명'], best_match_row['리명']
    
    return None, None, None, None, None

# --- 3. 테스트 케이스 동적 생성 (전략적 샘플링) ---
DTYPE_COLUMN = 'dm_ntype'
MESSAGE_COLUMN = 'message_content'
NUM_TOP_TYPES = 5
SAMPLES_PER_TYPE = 5

# 1. 재난 유형(dm_ntype) 개수 확인하여 상위 5개 선정
if DTYPE_COLUMN not in df_disaster.columns:
    print(f"오류: '{DTYPE_COLUMN}' 열이 없습니다.")
    exit()
top_types = df_disaster[DTYPE_COLUMN].value_counts().nlargest(NUM_TOP_TYPES).index.tolist()
print(f"\n✓ 상위 {NUM_TOP_TYPES}개 재난 유형 선정: {top_types}")

# 2. 각 유형별로 메시지(message_content) 5개씩 샘플링
if MESSAGE_COLUMN not in df_disaster.columns:
    print(f"오류: '{MESSAGE_COLUMN}' 열이 없습니다.")
    exit()

test_messages = []
for dtype in top_types:
    type_messages = df_disaster[df_disaster[DTYPE_COLUMN] == dtype][MESSAGE_COLUMN].dropna()
    num_samples = min(SAMPLES_PER_TYPE, len(type_messages))
    if num_samples > 0:
        samples = type_messages.sample(n=num_samples, random_state=1).tolist()
        test_messages.extend(samples)

print(f"✓ 총 {len(test_messages)}개의 테스트 케이스를 전략적으로 추출했습니다.")


print("\n--- 테스트 실행 ---")
# --- 4. 테스트 루프 실행 ---
for i, message in enumerate(test_messages):
    message = str(message)
    print(f"\n[테스트 {i+1}]")
    print(f"입력 문자: {message[:]}...")
    
    bjd_code, sido, sigungu, eupmyeon, ri = find_location_code_hierarchical(message)
    if sido is None and sigungu is None and eupmyeon is None and ri is None and bjd_code is None:
        continue
    else:
        print(f"  - 감지된 시도: {sido}")
        print(f"  - 감지된 시군구: {sigungu}")
        print(f"  - 감지된 읍면동: {eupmyeon}")
        print(f"  - 감지된 리: {ri}")
        print(f"  - 최종 법정동코드: {bjd_code}")

print("\n--- 테스트가 모두 완료되었습니다. ---")