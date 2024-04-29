import sys
import pandas as pd
from pandas.errors import ParserError

digit_count = 10

def read_csv(file_path):
    try:
        data = pd.read_csv(file_path)
    except ParserError as e:
        print("파일 데이터 형식문제입니다.\n대부분 해당 csv파일을 MS Excel로 연 뒤, 저장만 해줘도 형식에 맞게 데이터 형식을 맞춰주므로, 해당 방법을 시도해보는 것을 권장합니다.")
        exit(1)
    except:
        print("No such file : ", file_path)
        exit(1)
    return data

def find_actual_duration(data):
    
    # '참석자 보고서' 열에서 'SW프로그래밍의기초 09분반'을 찾아 '실제 기간(분)'을 반환
    class_row = data[data['참석자 보고서'].str.contains('SW프로그래밍의기초 09분반', na=False)]
    actual_duration = class_row.iloc[0]['Unnamed: 3'] if not class_row.empty else "클래스 정보를 찾을 수 없습니다."
    
    return actual_duration


from datetime import datetime

def extract_report_date(data):
    report_row = data[data['참석자 보고서'].str.contains('생성된 보고서:', na=False)]
    if not report_row.empty:
        report_date = report_row.iloc[0]['Unnamed: 1']
        return _format_date(report_date.strip())
    else:
        return "날짜 정보를 찾을 수 없습니다."

def _format_date(data):
    date_part = data.split(' ')[0:3]
    data = ' '.join(date_part)
    date_obj = datetime.strptime(data, '%m월 %d, %Y')
    formatted_date = date_obj.strftime('%Y년 %m월 %d일')
    return formatted_date

def process_attendee_details_from_data(data):
    """
    '참석자 세부 정보' 이후의 데이터를 새로운 헤더로 설정하고 필요한 데이터만 추출하는 함수.
    
    Parameters:
    - data (pd.DataFrame): 웨비나 참석자 보고서 데이터를 담고 있는 DataFrame 객체.
    
    Returns:
    - data (pd.DataFrame): '사용자 이름(원래 이름)', '세션 기간(분)' 열만 포함하는 DataFrame.
    """
    # '참석자 세부 정보' 행 인덱스 찾기
    header_row_idx = data.index[data.apply(lambda x: x.str.contains('참석자 세부 정보', na=False)).any(axis=1)][0] + 1
    
    new_header = data.iloc[header_row_idx]
    data = data[header_row_idx + 1:]  # 헤더 다음 데이터부터 선택
    data.columns = new_header  # 새 헤더 설정
    
    # 필요한 열만 선택
    data = data[['사용자 이름(원래 이름)', '세션 기간(분)']].copy()
    data.reset_index(drop=True, inplace=True)
    
    return data

def aggregate_session_times(data):
    """
    '사용자 이름(원래 이름)'이 같은 행을 합치고, '세션 기간(분)'을 더하는 함수.
    
    Parameters:
    - data (pd.DataFrame): 처리된 웨비나 참석자 데이터를 담고 있는 DataFrame 객체.
    
    Returns:
    - data (pd.DataFrame): 사용자 이름 별로 합쳐진 세션 기간을 담고 있는 데이터프레임.
    """
    # '세션 기간(분)'을 수치 데이터로 변환
    data['세션 기간(분)'] = pd.to_numeric(data['세션 기간(분)'], errors='coerce')
    
    # '사용자 이름(원래 이름)'으로 그룹화하고 '세션 기간(분)'을 합함
    data = data.groupby('사용자 이름(원래 이름)', as_index=False)['세션 기간(분)'].sum()
    
    return data


def reorder_and_clean_name_number(data):
    global digit_count
    """
    '사용자 이름(원래 이름)'에서 이름과 숫자(10자리)의 순서를 바꾸고, 구분 문자를 삭제하는 함수.
    
    Parameters:
    - data (pd.DataFrame): 집계된 웨비나 참석자 데이터를 담고 있는 DataFrame 객체.
    
    Returns:
    - cleaned_data (pd.DataFrame): 이름과 숫자 순서가 바뀌고, 구분 문자가 삭제된 '사용자 이름(원래 이름)'을 갖는 데이터프레임.
    """
    def clean_and_reorder_name(name):
        global digit_count
        # 숫자와 이름 사이의 구분 문자(공백, '/', '-', '_') 제거 후 순서 바꾸기
        import re
        # 패턴: 이름 [구분 문자] 숫자(10자리)
        pattern_text = fr"([^\d/\-_]+)[/\-_ ]*(\d{{{digit_count}}})"
        pattern = re.compile(pattern_text)
        match = pattern.search(name)
        if match:
            # 순서 바꾸기 (숫자, 이름) 및 구분 문자 삭제
            # print(f"{match.group(2)} {match.group(1)}")
            return f"{match.group(2)} {match.group(1)}"
        else:
            # 패턴에 맞지 않는 경우, 원래 이름 반환
            return name
    
    # 이름과 숫자 순서 바꾸기 및 구분 문자 제거
    cleaned_data = data.copy()

    cleaned_data['사용자 이름(원래 이름)'] = data['사용자 이름(원래 이름)'].apply(clean_and_reorder_name)
    
    return cleaned_data

def remove_separators_from_names(data):
    """
    '사용자 이름(원래 이름)'에서 구분 문자(공백, '/', '-', '_')를 제거하는 함수.
    
    Parameters:
    - data (pd.DataFrame): 데이터를 담고 있는 DataFrame 객체.
    
    Returns:
    - modified_data (pd.DataFrame): 구분 문자가 제거된 '사용자 이름(원래 이름)'을 갖는 데이터프레임.
    """
    def remove_separators(name):
        import re
        # 구분 문자(공백, '/', '-', '_') 제거
        cleaned_name = re.sub(r"[\s\/\-_]", "", name)
        return cleaned_name
    
    modified_data = data.copy()
    modified_data['사용자 이름(원래 이름)'] = data['사용자 이름(원래 이름)'].apply(remove_separators)
    
    return modified_data

import re
def filter_and_return_special_cases_with_reason(data):
    global digit_count
    """
    조건(이름만 있거나, 숫자만 있거나, 숫자와 이름 형식이지만 숫자가 10자리가 아닌 경우)에 해당하는 행을 찾아서
    해당 '사용자 이름(원래 이름)', '세션 기간(분)', 그리고 원인을 리스트로 반환하고,
    이 행들을 원본 데이터프레임에서 제거한 후 수정된 데이터프레임을 반환하는 함수.

    Parameters:
    data (pd.DataFrame): 입력 데이터프레임

    Returns:
    filtered_data (pd.DataFrame): 수정된 데이터프레임
    special_cases_with_reason (list): 조건에 해당하는 별도 처리가 필요한 데이터 리스트 (이름, 세션 기간(분), 원인)
    """
    special_cases_with_reason = []  # 조건에 맞는 경우들의 리스트 (이름, 세션 기간(분), 원인)

    # 조건에 따른 필터링을 위한 함수
    def filter_condition(name):
        global digit_count
        pattern = r'[\u1100-\u1112\u1161-\u1175\u11A8-\u11C2\u3131-\u314E\u3165-\u318E]' # 한글 자모음 유니코드
        pattern_text = fr"\d{{{digit_count}}}"
        if re.search(pattern, name):
            return "올바르지 않은 문자"
        elif re.fullmatch(r'[가-힣a-zA-Z]+', name):
            return "이름만 있음"
        elif re.fullmatch(r'\d+', name):
            return "숫자만 있음"
        elif not re.match(pattern_text, name):
            return "학번이름이 올바르지 않음."
        elif re.search(r'(?:(?:\D+\d+)|(\d+\D+))\d*', name) and not re.match(pattern_text, name):
            return "숫자가 10자리가 아닌 숫자+이름 형식"
        return None

    # 조건에 맞는 행 찾기
    for index, row in data.iterrows():
        name = row['사용자 이름(원래 이름)']
        reason = filter_condition(name)
        if reason:
            special_cases_with_reason.append((name, row['세션 기간(분)'], reason))

    # 조건에 맞는 행 삭제
    filtered_data = data[~data['사용자 이름(원래 이름)'].apply(lambda name: bool(filter_condition(name)))]
    
    return filtered_data, special_cases_with_reason

def merge_special_cases_with_main_data_updated(filtered_df, special_cases):
    """
    스페셜 케이스와 필터링된 데이터를 병합하는 함수. 스페셜 케이스에서 이름 또는 숫자가 필터링된 데이터 중
    '사용자 이름(원래 이름)'에 단 하나만 존재할 경우, 해당 스페셜 케이스의 세션 기간(분)을 필터링된 데이터의
    매칭되는 행에 더하고, 스페셜 케이스에서 해당 항목을 삭제한다. 매칭이 두 개 이상이거나 없는 경우 원인을 업데이트하는 함수.

    Parameters:
    filtered_df (pd.DataFrame): 필터링된 데이터프레임
    special_cases (list): 스페셜 케이스 리스트 (이름, 세션 기간(분), 원인)

    Returns:
    filtered_df (d.DataFrame): 수정된 데이터프레임
    remaining_special_cases (list): 수정된 스페셜 케이스 리스트
    """
    remaining_special_cases = special_cases.copy()
    for name, duration, reason in special_cases:
        matches = filtered_df[filtered_df['사용자 이름(원래 이름)'].str.contains(name, regex=False)]
        
        # 단 하나의 매칭만 존재하는 경우
        if len(matches) == 1:
            match_index = matches.index[0]
            filtered_df.at[match_index, '세션 기간(분)'] = str(int(filtered_df.at[match_index, '세션 기간(분)']) + int(duration))
            remaining_special_cases.remove((name, duration, reason))
        elif len(matches) > 1:
            # 매칭되는 이름이 여러 개인 경우, 원인 업데이트
            for i, sc in enumerate(remaining_special_cases):
                if sc[0] == name:
                    remaining_special_cases[i] = (name, duration, "매칭되는 이름이 여러 개입니다")
        else:
            # 매칭되는 결과가 없는 경우, 원인 업데이트
            for i, sc in enumerate(remaining_special_cases):
                if sc[0] == name:
                    remaining_special_cases[i] = (name, duration, "매칭되는 결과가 없습니다")
    
    return filtered_df, remaining_special_cases

def add_space_between_id_and_name(data):
    global digit_count
    """
    '사용자 이름(원래 이름)' 열에서 학번(숫자 10자리)과 이름 사이에 공백을 추가하는 함수.

    Parameters:
    dataframe (pd.DataFrame): 입력 데이터프레임

    Returns:
    pd.DataFrame: 수정된 데이터프레임
    """
    pattern_text = fr'(\d{{{digit_count}}})([가-힣a-zA-Z]+)'
    # 학번과 이름 사이에 공백 추가
    data['사용자 이름(원래 이름)'] = data['사용자 이름(원래 이름)'].apply(lambda x: re.sub(pattern_text, r'\1 \2', x))
    
    return data

def save_to_excel(final_df, special_cases_with_reason, file_path):
    global digit_count
    """
    '사용자 이름(원래 이름)'을 학번과 이름으로 분리하여 엑셀 파일로 저장하는 함수.
    
    Parameters:
    final_df (pd.DataFrame): 정상 데이터가 포함된 데이터프레임
    special_cases_with_reason (list): 스페셜 케이스와 그 원인이 포함된 리스트
    file_path (str): 저장할 엑셀 파일의 경로
    """
    pattern_text = fr'(?P<학번>\d{{{digit_count}}})?\s?(?P<이름>[가-힣a-zA-Z]+)'

    # 학번과 이름 분리
    split_data = final_df['사용자 이름(원래 이름)'].str.extract(pattern_text)
    # 닉네임은 원래 이름에 유지
    split_data['닉네임'] = final_df['사용자 이름(원래 이름)']
    # 세션 기간(분) 추가
    split_data['세션 기간(분)'] = final_df['세션 기간(분)']
    # 출석 상태 추가
    split_data['출석 상태'] = final_df['출석 상태']
    
    
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # "정상 처리 리스트" 텍스트를 첫 번째 행에 추가
        pd.DataFrame(["정상 처리 리스트"]).to_excel(writer, index=False, header=False, startrow=0, startcol=0)
        
        # 두 번째 행부터 정상 데이터 추가
        split_data.to_excel(writer, index=False, header=True, startrow=1, startcol=0)
        
        # 미처리 리스트 시작 행 계산
        special_cases_start_row = len(split_data) + 3  # 정상 데이터 행 수 + 2(텍스트 행 + 공백 행)
        
        # "미처리 리스트" 텍스트 추가
        pd.DataFrame(["미처리 리스트"]).to_excel(writer, index=False, header=False, startrow=special_cases_start_row, startcol=0)
        
        # 스페셜 케이스 데이터 추가
        pd.DataFrame(special_cases_with_reason, columns=['사용자 이름(원래 이름)', '세션 기간(분)', '원인']).to_excel(writer, index=False, header=True, startrow=special_cases_start_row + 1, startcol=0)


def add_attendance_status_based_on_threshold(data, threshold, class_duration):
    """
    '세션 기간(분)'이 수업시간의 임계값 퍼센트 이상을 만족하는지 여부에 따라 '출석' 또는 '시청 시간 부족'을 추가하는 함수.

    Parameters:
    data (pd.DataFrame): 정상 처리된 데이터가 포함된 데이터프레임
    threshold (int): 임계값 퍼센트 (0~100 사이의 정수)
    class_duration (int): 수업시간 (분)

    Returns:
    data (pd.DataFrame): '출석 상태' 열이 추가된 데이터프레임
    """
    # '출석 상태' 열 계산 및 추가
    data['출석 상태'] = data['세션 기간(분)'].apply(lambda x: '출석' if int(x) >= int(float(class_duration) * (threshold / 100)) else '시청 시간 부족')
    
    return data


import argparse

def main():
    global digit_count
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='Process some integers.')

    # '-path' 인자 추가: 파일 경로, 기본값 설정
    parser.add_argument('-p', '--path', type=str, help='Path to the csv file', default=None, required=True, metavar='path of csv file')

    # '-t' 인자 추가: 임계값, 기본값 설정
    parser.add_argument('-t', '--threshold', type=int, help='(Optional) Minimum required attendance percentage (default : 70)', default=70, metavar='threshold')

    # '-l' 인자 추가: 학번(고유번호) 길이 설정
    parser.add_argument('-l', '--length', type=int, help='(Optional) Minimum required attendance percentage (default : 70)', default=70, metavar='threshold')

    # 인자 파싱
    args = parser.parse_args()
    threshold = 0
    # 인자 존재 여부 확인
    if args.path is None:
        print('No file path provided.')
        exit(1)
       
    if args.threshold is None:
        pass
    elif int(args.threshold) > 100 or int(args.threshold) < 0:
        print("usage: main.py [-h] -p PATH -t T\nmain.py: error: Invalid literal for int() between 0 and 100 --threshold / -t")
        exit(1)
    else:
        threshold = args.threshold

    if args.length is not None:
        digit_count = int(args.length)

    data = read_csv(file_path=args.path)
    # 기본 정보 
    duration = find_actual_duration(data)
    report_date = extract_report_date(data)
    # 데이터 전처리
    data = process_attendee_details_from_data(data) # 참석자 필터링
    data = aggregate_session_times(data) # 동일 행 병합
    data = reorder_and_clean_name_number(data) # 학번 이름 순으로 재정렬
    data = remove_separators_from_names(data) # 공백 삭제
    data = aggregate_session_times(data) # 동일 행 병합
    data, filtered_data = filter_and_return_special_cases_with_reason(data) # 정삭이름이 아닌 경우 필터링
    data, refiltered_data = merge_special_cases_with_main_data_updated(data, filtered_data)
    data = add_space_between_id_and_name(data)
    # print(data)
    data = add_attendance_status_based_on_threshold(data, threshold=threshold, class_duration=duration)
    print(data)
    save_to_excel(data, refiltered_data, './'+report_date+' output.xlsx')
        

if __name__ == "__main__":
    main()