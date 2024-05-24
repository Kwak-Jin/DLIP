import os
import re
def replace_in_file(file_path, old_string: None, new_string):
    """
    파일에서 특정 문자열을 다른 문자열로 대체하는 함수.

    Parameters:
    file_path (str): 파일 경로
    old_string (str): 대체할 문자열
    new_string (str): 새로운 문자열

    Returns:
    None
    """
    try:
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = file.read()

        # 문자열 대체
        new_data = file_data.replace(old_string, new_string)

        # 파일 쓰기
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_data)

        print(f"'{old_string}'이(가) '{new_string}'으로 대체되었습니다.")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except IOError:
        print(f"파일을 읽거나 쓸 수 없습니다: {file_path}")
# 디렉토리 경로를 지정합니다.
directory_path = "./obj"

replace_in_file(directory_path,)