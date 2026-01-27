import cv2
import numpy as np
import os
import sys

# 상위 디렉토리의 charuco_config.py를 import하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from charuco_config import (
    CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
    CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH
)

# ==========================================
# 1. 설정 (사용자 최적화 값 유지)
# ==========================================
IMAGES_DIR = "test_scene/rgb"
PROCESSED_DIR = "test_scene/rgb_solid"
DEBUG_DIR = "test/debug_solid_color"

# 마진 비율 (넓게 잡음)
MARGIN_RATIO = 0.6
# 마스크 확장 (흰색 테두리 완전 제거)
DILATION_ITERATIONS = 20
# 경계선 부드럽게 (홀수)
BLUR_KERNEL_SIZE = (21, 21)

for d in [PROCESSED_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# Detector 설정
try:
    charuco_board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, aruco_dict
    )
    aruco_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
    USE_CHARUCO_DETECTOR = True
    print("✅ OpenCV 4.7+ CharucoDetector 사용")
except AttributeError:
    charuco_board = cv2.aruco.CharucoBoard_create(
        CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
        CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, aruco_dict
    )
    aruco_params = cv2.aruco.DetectorParameters_create()
    charuco_detector = None
    USE_CHARUCO_DETECTOR = False
    print("⚠️ 구버전 OpenCV 사용")

def get_smart_grabcut_mask(image, corners):
    """
    (기존 유지) GrabCut으로 넓은 마스크 생성
    """
    h, w = image.shape[:2]
    all_points = np.vstack([c.reshape(-1, 2) for c in corners])
    
    x_min, y_min = np.min(all_points, axis=0).astype(int)
    x_max, y_max = np.max(all_points, axis=0).astype(int)
    
    board_w = x_max - x_min
    board_h = y_max - y_min
    
    # 동적 마진
    margin_x = int(board_w * MARGIN_RATIO)
    margin_y = int(board_h * MARGIN_RATIO)
    
    rect_x1 = max(0, x_min - margin_x)
    rect_y1 = max(0, y_min - margin_y)
    rect_x2 = min(w, x_max + margin_x)
    rect_y2 = min(h, y_max + margin_y)
    
    # GrabCut 초기화
    mask_gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    mask_gc[rect_y1:rect_y2, rect_x1:rect_x2] = cv2.GC_PR_FGD
    for c in corners:
        pt = c.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask_gc, [pt], cv2.GC_FGD)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image, mask_gc, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
    except:
        return np.zeros((h, w), dtype=np.uint8)

    mask_final = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')

    # 구멍 메우기 & 노이즈 제거
    kernel_close = np.ones((7, 7), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask_final)
        cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
        mask_final = mask_clean

    if DILATION_ITERATIONS > 0:
        kernel_dilate = np.ones((3, 3), np.uint8)
        mask_final = cv2.dilate(mask_final, kernel_dilate, iterations=DILATION_ITERATIONS)

    return mask_final

def fill_with_solid_color_sample(image, mask, corners):
    """
    [요청 반영] 텍스처 복사 X, 노이즈 추가 X
    마스크 바깥 영역에서 여러 포인트를 샘플링하여 중앙값 색상으로 채움.
    Returns: (result_image, sample_point (sx, sy))
    """
    if not np.any(mask): return image, None
    h, w = image.shape[:2]

    # 1. 보드 오른쪽에서 색상 샘플링
    all_pts = np.vstack([c[0] for c in corners])
    max_x = int(np.max(all_pts[:, 0]))
    min_y = int(np.min(all_pts[:, 1]))
    max_y = int(np.max(all_pts[:, 1]))
    board_w = max_x - int(np.min(all_pts[:, 0]))

    # 오른쪽 상단 지점에서 샘플링
    board_h = max_y - min_y
    sample_offset = int(board_w * 0.6)
    sx = min(w - 1, max_x + sample_offset)
    sy = max(0, min_y - int(board_h * 0.3))  # 위로 이동

    # 마스크 바깥인지 확인 후 샘플링
    if mask[sy, sx] == 0:
        sample_color = image[sy, sx].astype(np.float32)
    else:
        # 폴백: 정중앙
        sx, sy = w // 2, h // 2
        sample_color = image[sy, sx].astype(np.float32)

    # 3. 단색 레이어 생성 (전체 이미지 크기, 노이즈 없음!)
    solid_layer = np.full((h, w, 3), sample_color, dtype=np.float32)

    # 4. 부드러운 블렌딩 (Soft Edge)
    # 마스크 경계를 흐릿하게 만듦 (0.0 ~ 1.0 알파 채널)
    mask_soft = cv2.GaussianBlur(mask, BLUR_KERNEL_SIZE, 0)
    alpha = cv2.merge([mask_soft, mask_soft, mask_soft]).astype(np.float32) / 255.0

    # 5. 합성 (Alpha Blending)
    # 마스크 영역은 단색 레이어, 나머지는 원본 이미지 사용
    foreground = solid_layer * alpha
    background = image.astype(np.float32) * (1.0 - alpha)

    result = cv2.add(foreground, background).astype(np.uint8)

    return result, (sx, sy)

# ==========================================
# 실행
# ==========================================
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg'))]
print(f"📊 Processing {len(image_files)} images (Solid Color Fill)...")

for idx, img_name in enumerate(image_files):
    img_path = os.path.join(IMAGES_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None: continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if USE_CHARUCO_DETECTOR:
        corners, ids, _, _ = charuco_detector.detectBoard(gray)
        if corners is None or len(corners) == 0:
             corners, ids, _ = cv2.aruco.ArucoDetector(aruco_dict, aruco_params).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is None or len(corners) == 0: continue

    # 1. GrabCut 마스크 (넓게)
    board_mask = get_smart_grabcut_mask(image, corners)

    # 2. 단색 채우기 (노이즈/텍스처 없음)
    image_clean, sample_pt = fill_with_solid_color_sample(image, board_mask, corners)

    cv2.imwrite(os.path.join(PROCESSED_DIR, f"{img_name.split('.')[0]}.png"), image_clean)

    if idx % 10 == 0:
        # 원본 이미지에 샘플링 위치 표시
        debug = image.copy()
        if sample_pt:
            sx, sy = sample_pt
            cv2.circle(debug, (sx, sy), 10, (0, 0, 255), -1)  # 빨간 원
            cv2.putText(debug, f"Sample: ({sx}, {sy})", (sx + 15, sy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(debug, "Method: Solid Color Sample", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_solid_{img_name}"), debug)
        print(f"✅ {img_name} Done (sample at {sample_pt})")

print("\n🎉 완료! 요청하신 대로 '단색'으로 깔끔하게 덮었습니다.")