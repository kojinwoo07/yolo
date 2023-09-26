
#  NVIDIA GPU 사용가능한지 여부를 판단
!nvidia-smi  
# 구글 드라이브의 파일을 저장하고 불러오기 위한 접근성 할당
%cd /content/gdrive/MyDrive
Creating Image Directory 
Creating Custom.names file 
Creating Train and Text files
Creating Backup directory
Creating YOLO data file in the custome_data  
# !git clone 명령어를 통해 dark network을 다운로드 하고 생성해줍니다.
!git clone https://github.com/AlexeyAB/darknet
# 현재 작업중인 directory를 darknet으로 변경
%cd darknet


# Makefile에서 GPU와 OpenCV가 사용가등하도록 변경
%cd /content/gdrive/MyDrive/darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
# Darknet 프레임워크의 루트 디렉토리에 있는 Makefile을 가리키는 코드
"/content/darknet/Makefile"

# 현재 디렉토리에 있는 Makefile을 사용하여 프로젝트를 빌드
!make

# 미리 학습된 가중치 모델 다운로드
!wget https://pjreddie.com/media/files/darknet53.conv.74

# 학습 시작
!./darknet detector train /content/gdrive/MyDrive/custom_data/detector.data /content/gdrive/MyDrive/custom_data/cfg/yolov3-custom.cfg darknet53.conv.74 -dont_show
