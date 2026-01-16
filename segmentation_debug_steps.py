import cv2
import numpy as np
import os

# 테스트할 이미지 선택
IMAGE_PATH = "dataset/final_solid_color/final_image_20260112_132024_029.png"
OUTPUT_DIR = "dataset/debug_steps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 설정
COLOR_TOLERANCE = (5, 5, 5)
CORNER_DIFF_THRESH = 60.0
CORNER_BOX_SIZE = 40

def calculate_color_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

# 이미지 로드
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Cannot load {IMAGE_PATH}")
    exit(1)

h, w = image.shape[:2]
print(f"이미지 크기: {w} x {h}")

# ========================================
# Step 1: 방파제 생성
# ========================================
print("\n=== Step 1: 방파제 생성 ===")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Canny 엣지 검출
edges = cv2.Canny(gray, 20, 60)
print(f"Canny 엣지 픽셀 수: {np.sum(edges > 0)}")

debug1a = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.putText(debug1a, "Canny Edges (20/60)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step1a_canny.png", debug1a)

# Closing
kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_connect)
print(f"Closing 후 픽셀 수: {np.sum(closed_edges > 0)}")

debug1b = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2BGR)
cv2.putText(debug1b, "After Closing", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step1b_closing.png", debug1b)

# Dilation
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
thick_edges = cv2.dilate(closed_edges, kernel_dilate, iterations=3)
print(f"Dilation 후 픽셀 수: {np.sum(thick_edges > 0)}")

debug1c = image.copy()
debug1c[thick_edges > 0] = (0, 0, 255)
cv2.putText(debug1c, "Barrier (Red)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step1c_barrier.png", debug1c)

edge_barrier = thick_edges

# ========================================
# Step 2: 시작점 찾기
# ========================================
print("\n=== Step 2: 시작점 찾기 ===")
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cy, cx = h // 2, w // 2
center_roi = lab_image[cy-10:cy+10, cx-10:cx+10]
center_mean_lab = cv2.mean(center_roi)[:3]

corners_map = {
    "TL": (0, 0), "TR": (0, w-CORNER_BOX_SIZE),
    "BL": (h-CORNER_BOX_SIZE, 0), "BR": (h-CORNER_BOX_SIZE, w-CORNER_BOX_SIZE)
}
object_seeds = [(w//2, h//2)]

debug2 = image.copy()
cv2.circle(debug2, (cx, cy), 30, (255, 0, 0), -1)

for name, (py, px) in corners_map.items():
    roi = lab_image[py:py+CORNER_BOX_SIZE, px:px+CORNER_BOX_SIZE]
    corner_mean = cv2.mean(roi)[:3]
    dist = calculate_color_distance(center_mean_lab, corner_mean)

    if dist < CORNER_DIFF_THRESH:
        seed_point = (px + CORNER_BOX_SIZE//2, py + CORNER_BOX_SIZE//2)
        object_seeds.append(seed_point)
        cv2.circle(debug2, seed_point, 20, (0, 255, 0), -1)
        print(f"  {name}: OK")

cv2.putText(debug2, f"Seeds: {len(object_seeds)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step2_seeds.png", debug2)
print(f"시작점: {len(object_seeds)}개 - {object_seeds}")

# ========================================
# Step 3: FloodFill 마스크 설정
# ========================================
print("\n=== Step 3: FloodFill 마스크 설정 ===")
h_mask, w_mask = h + 2, w + 2
mask_flood = np.zeros((h_mask, w_mask), np.uint8)

# 방파제를 마스크에 설정
mask_flood[1:h+1, 1:w+1] = edge_barrier
print(f"마스크에 방파제 설정: {np.sum(mask_flood > 0)} 픽셀")

# 방파제를 1로 변환
mask_flood[mask_flood > 0] = 1
print(f"방파제를 1로 변환: {np.sum(mask_flood == 1)} 픽셀")

debug3 = np.zeros((h, w, 3), dtype=np.uint8)
debug3[mask_flood[1:h+1, 1:w+1] == 1] = (0, 0, 255)
cv2.putText(debug3, "Barrier=1 (Red)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step3_mask_barrier.png", debug3)

# ========================================
# Step 4: FloodFill 실행
# ========================================
print("\n=== Step 4: FloodFill 실행 ===")
for i, seed in enumerate(object_seeds):
    print(f"  시작점 {i+1}: {seed}")
    try:
        img_copy = image.copy()
        num_filled = cv2.floodFill(img_copy, mask_flood, seed, (255, 255, 255),
                                   COLOR_TOLERANCE, COLOR_TOLERANCE,
                                   flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
        print(f"    채워진 픽셀: {num_filled[0]}")
    except Exception as e:
        print(f"    Error: {e}")

# 마스크 상태 확인
print(f"\n마스크 값 분포:")
print(f"  0 (빈 공간): {np.sum(mask_flood == 0)}")
print(f"  1 (방파제): {np.sum(mask_flood == 1)}")
print(f"  255 (채워진 영역): {np.sum(mask_flood == 255)}")

# ========================================
# Step 5: 결과 추출
# ========================================
print("\n=== Step 5: 결과 추출 ===")
result_mask = (mask_flood[1:h+1, 1:w+1] == 255).astype(np.uint8) * 255
filled_pixels = np.sum(result_mask > 0)
print(f"최종 채워진 픽셀: {filled_pixels} / {h*w} ({filled_pixels/(h*w)*100:.1f}%)")

debug5a = image.copy()
overlay = image.copy()
overlay[result_mask > 0] = (0, 255, 0)
debug5a = cv2.addWeighted(debug5a, 0.5, overlay, 0.5, 0)
cv2.putText(debug5a, "FloodFill Result (Green)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step5a_flood_result.png", debug5a)

# 방파제 + FloodFill 같이 표시
debug5b = image.copy()
debug5b[edge_barrier > 0] = (0, 0, 255)
overlay = debug5b.copy()
overlay[result_mask > 0] = (0, 255, 0)
debug5b = cv2.addWeighted(debug5b, 0.7, overlay, 0.3, 0)
cv2.putText(debug5b, "Red=Barrier, Green=Flood", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.imwrite(f"{OUTPUT_DIR}/step5b_combined.png", debug5b)

# ========================================
# Step 6: 윤곽선 및 다각형
# ========================================
print("\n=== Step 6: 윤곽선 및 다각형 ===")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
mask_closed = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"윤곽선 개수: {len(contours)}")

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    print(f"가장 큰 윤곽선 점 개수: {len(largest_contour)}")

    # Convex hull로 외곽선 펴기 (안으로 찌그러진 것 방지)
    hull = cv2.convexHull(largest_contour)
    print(f"Convex hull 점 개수: {len(hull)}")

    peri = cv2.arcLength(hull, True)
    found_polys = {}

    # 더 세밀한 epsilon 범위 (0.003~0.04)
    for eps_rate in np.linspace(0.003, 0.04, 150):
        epsilon = eps_rate * peri
        approx = cv2.approxPolyDP(hull, epsilon, True)
        pts = len(approx)
        if pts in [4, 5, 6]:
            if pts not in found_polys:
                found_polys[pts] = (approx, eps_rate)
                print(f"  {pts}각형 발견 (eps_rate={eps_rate:.3f})")

    for num_pts in [4, 5, 6]:
        if num_pts in found_polys:
            poly, eps_rate = found_polys[num_pts]
            debug6 = image.copy()
            cv2.drawContours(debug6, [poly], -1, (0, 255, 0), 5)

            for i, pt in enumerate(poly):
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(debug6, (x, y), 25, (0, 0, 255), -1)
                cv2.circle(debug6, (x, y), 27, (255, 255, 255), 3)
                cv2.putText(debug6, str(i+1), (x-12, y+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.putText(debug6, f"{num_pts}-gon", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imwrite(f"{OUTPUT_DIR}/step6_{num_pts}gon.png", debug6)

print(f"\n✅ 완료! {OUTPUT_DIR}/ 에서 확인")
