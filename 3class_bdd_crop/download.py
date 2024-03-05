import shutil
import os
import urllib.request



def create_list_file(directory_path, output_file_path):
    """
    특정 디렉토리 안에 있는 .jpg 파일들의 경로를 텍스트 파일로 저장하는 함수.

    Parameters:
    - directory_path: 파일들이 있는 디렉토리 경로
    - output_file_path: 파일 경로들을 저장할 텍스트 파일 경로
    """
    with open(file=output_file_path, mode='w') as file:
        # 디렉토리 안에 있는 모든 파일들의 경로 가져오기
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if file_name.lower().endswith('.jpg'):
                    file_path = os.path.join(root, file_name)
                    file.write(file_path + '\n')

    # print(f"File paths written to {output_file_path}")

# USE
create_list_file(directory_path="/home/msis/Work/yolov2-pytorch/3class_bdd/train256", output_file_path="/home/msis/Work/yolov2-pytorch/3class_bdd/train256.txt")


def download_image_by_txtFile(textFile, targetDirectory):

    # train64.txt 파일을 읽어서 파일 경로들을 리스트에 저장
    with open(file=textFile, mode="r") as file:
        lines = file.readlines()

        for line in lines:
            file_path = line.strip()

            image_file = file_path
            shutil.copy(src=image_file, dst=targetDirectory)

            label_file = file_path.replace(".jpg", ".txt")
            shutil.copy(src=label_file, dst=targetDirectory)

    print("복사가 완료되었습니다.")

# USE
# download_image_by_txtFile(textFile="Bdd_uncleaned/3class_bdd/val64.txt", targetDirectory="./Bdd_uncleaned/3class_bdd/val64")