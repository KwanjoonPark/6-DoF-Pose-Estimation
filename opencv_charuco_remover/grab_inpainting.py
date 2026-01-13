import cv2
import numpy as np
import os

# ==========================================
# 1. 설정 (사용자 최적화 값 유지)
# ==========================================
IMAGES_DIR = "images"
PROCESSED_DIR = "dataset/final_stable_inpainting"
DEBUG_DIR = "dataset/debug_stable"

# 마진 비율 (0.45 = 45% 여유, 넓게 잡음)
MARGIN_RATIO = 0.45
# 마스크 확장 (흰색 테두리 완전 제거용, 중요!)
DILATION_ITERATIONS = 12
# 인페인팅 참조 반경 (주변 픽셀 탐색 범위)
INPAINT_RADIUS = 5

# ChArUco 파라미터
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 4
CHARUCO_SQUARE_LENGTH = 0.02
CHARUCO_MARKER_LENGTH = 0.015

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
    """ (기존 유지) GrabCut으로 넓고 정확한 마스크 생성 """
    h, w = image.shape[:2]
    all_points = np.vstack([c.reshape(-1, 2) for c in corners])
    
    x_min, y_min = np.min(all_points, axis=0).astype(int)
    x_max, y_max = np.max(all_points, axis=0).astype(int)
    
    board_w = x_max - x_min
    board_h = y_max - y_min
    
    margin_x = int(board_w * MARGIN_RATIO)
    margin_y = int(board_h * MARGIN_RATIO)
    
    rect_x1 = max(0, x_min - margin_x)
    rect_y1 = max(0, y_min - margin_y)
    rect_x2 = min(w, x_max + margin_x)
    rect_y2 = min(h, y_max + margin_y)
    
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

def fill_with_inpainting_telea(image, mask):
    """
    [복귀] 표준 인페인팅 알고리즘 사용
    - 줄무늬(Streaks)가 생기지 않음
    - 주변의 모든 방향에서 질감과 조명을 부드럽게 혼합하여 채움
    - 마스크가 정확하다면 가장 안정적인 결과물 제공
    """
    if not np.any(mask): return image
    
    # INPAINT_TELEA: 빠르고 자연스러운 결과물 생성
    # INPAINT_RADIUS: 주변 5픽셀 정보를 참조하여 채움
    result = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
    return result

# ==========================================
# 실행
# ==========================================
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg'))]
print(f"📊 Processing {len(image_files)} images (Stable Inpainting)...")

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

    # 1. 마스크 생성 (GrabCut으로 정확하게)
    board_mask = get_smart_grabcut_mask(image, corners)

    # 2. 채우기 (표준 인페인팅으로 안정적으로)
    image_clean = fill_with_inpainting_telea(image, board_mask)

    cv2.imwrite(os.path.join(PROCESSED_DIR, f"final_{img_name.split('.')[0]}.png"), image_clean)

    if idx % 10 == 0:
        debug = image_clean.copy()
        cv2.putText(debug, "Method: Telea Inpainting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_stable_{img_name}"), debug)
        print(f"✅ {img_name} Done")

print("\n🎉 완료! 줄무늬 없이 가장 안정적인 결과물로 복원되었습니다.")