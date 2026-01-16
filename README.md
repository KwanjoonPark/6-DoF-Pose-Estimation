# 6-DoF Pose Estimation

ChArUco 마커를 이용한 6자유도(6-DoF) 포즈 추정 시스템입니다. Intel RealSense 카메라와 ROS를 사용하여 실시간 또는 배치 처리로 3D 위치(x, y, z)와 회전(pitch, yaw, roll)을 계산합니다.

## 주요 기능

- ChArUco 마커 기반 6-DoF 포즈 추정
- ROS 실시간 처리 및 ROS bag 배치 처리
- Edge 기반 세그멘테이션 (FloodFill + 다각형 근사)
- ChArUco 마커 제거 (단색 채우기, 인페인팅)

## 요구 사항

### 하드웨어
- Intel RealSense D435/D415 카메라

### 소프트웨어
```bash
# Python 패키지
pip install opencv-python>=4.7 numpy

# ROS 패키지 (ROS Noetic/Melodic)
sudo apt install ros-noetic-cv-bridge ros-noetic-sensor-msgs
```

## 사용 방법

### 1. 실시간 포즈 추정

```bash
python charuco_pose_estimator.py
```
- 's' 키: 현재 프레임과 포즈 데이터 저장
- 'q' 키: 종료

### 2. ROS Bag 배치 처리

```bash
python rosbag_pose_estimator.py
```
bag 파일에서 모든 프레임을 추출하고 포즈를 계산합니다.

### 3. 마커 제거

```bash
# 단색으로 마커 영역 채우기
python opencv_charuco_remover/grab_solid.py

# 인페인팅으로 마커 영역 채우기
python opencv_charuco_remover/grab_inpainting.py
```

### 4. 세그멘테이션

```bash
# Edge 기반 세그멘테이션 (Canny + FloodFill + 다각형 근사)
python edge_based_segmentation.py
```

### 5. 디버깅

```bash
# edge_based_segmentation 단계별 시각화
python segmentation_debug_steps.py
```

## 파일 구조

```
├── charuco_pose_estimator.py       # 실시간 포즈 추정 (ROS 노드)
├── rosbag_pose_estimator.py        # ROS bag 배치 처리
├── edge_based_segmentation.py      # Edge 기반 세그멘테이션
├── segmentation_debug_steps.py     # 세그멘테이션 디버깅
└── opencv_charuco_remover/
    ├── grab_solid.py               # 단색 채우기로 마커 제거
    └── grab_inpainting.py          # 인페인팅으로 마커 제거
```

## ChArUco 보드 설정

| 파라미터 | 값 |
|---------|-----|
| 가로 사각형 수 | 5 |
| 세로 사각형 수 | 4 |
| 사각형 크기 | 2cm |
| 마커 크기 | 1.5cm |
| ArUco 사전 | DICT_4X4_250 |

## 출력 데이터

### CSV 형식
```csv
Timestamp, Image_File, Distance(cm), Pitch(deg), Yaw(deg), Roll(deg), X(cm), Y(cm), Z(cm)
```

### 출력 디렉토리
```
├── images/                          # RGB 이미지
├── depth/                           # Depth 이미지
├── self_pose.csv                    # 포즈 데이터
└── dataset/
    ├── final_solid_color/           # 단색 채우기 결과
    ├── final_stable_inpainting/     # 인페인팅 결과
    └── inpainting_background_off/   # 배경 제거 결과
```

## 알고리즘 개요

### 포즈 추정
1. ChArUco 코너 검출 (서브픽셀 정밀도)
2. `estimatePoseCharucoBoard()`로 PnP 풀이
3. 회전 행렬 → 오일러 각도 변환 (ZYX 순서)
4. 오프셋 변환 적용 (마커 → 목표 지점)

### 마커 제거 (grab_solid.py)
1. GrabCut으로 마커 영역 마스크 생성
2. 마스크 외부에서 단일 색상 샘플링
3. 단색으로 마스크 영역 채우기
4. Gaussian Blur로 경계 부드럽게 블렌딩

### 마커 제거 (grab_inpainting.py)
1. GrabCut으로 마커 영역 마스크 생성
2. OpenCV `inpaint()` (TELEA 알고리즘)으로 주변 색상 참조하여 채우기

### Edge 기반 세그멘테이션
1. Canny 엣지 검출 → Morphology Closing → Dilation으로 방파제 생성
2. 중심점 + 코너에서 FloodFill 시작점 결정 (LAB 색공간 비교)
3. FloodFill로 객체 영역 채우기 (방파제가 경계 역할)
4. Convex Hull → 다각형 근사 (4/5/6각형 우선순위)
