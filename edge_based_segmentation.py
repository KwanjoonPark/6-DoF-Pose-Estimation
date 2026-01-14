import cv2
import numpy as np
import os

# ==========================================
# Configuration
# ==========================================
IMAGES_DIR = "dataset/final_stable_inpainting"
PROCESSED_DIR = "dataset/inpainting_background_off"
DEBUG_DIR = "dataset/inpainting_debug_backgroud_off"

# [Config] Color similarity (FloodFill)
COLOR_TOLERANCE = (5, 5, 5)  # 9 -> 5 (more conservative)

# [Config] Corner similarity
CORNER_DIFF_THRESH = 60.0
CORNER_BOX_SIZE = 40

# [Config] Barrier - detect strong edges only
CANNY_THRESH_1 = 50   # Set high
CANNY_THRESH_2 = 150  # Set high

for d in [PROCESSED_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)

def calculate_color_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

def get_edge_barrier(image):
    """Edge detection - used as barrier"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection (low threshold - more edges)
    edges = cv2.Canny(gray, 20, 60)

    # Connect broken lines
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_connect)

    # Thicken barrier
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thick_edges = cv2.dilate(closed_edges, kernel_dilate, iterations=3)

    return thick_edges

def get_floodfill_mask(image, seeds, tolerance):
    """Generate object region mask using FloodFill (with barrier)"""
    h, w = image.shape[:2]

    # Create barrier
    edge_barrier = get_edge_barrier(image)

    # Mask for FloodFill (including barrier)
    # Important: mask size should be 2 pixels larger than image
    h_mask, w_mask = h + 2, w + 2
    mask_flood = np.zeros((h_mask, w_mask), np.uint8)

    # Set barrier in mask (255 = impassable wall)
    # Non-zero values in FloodFill mask are treated as already "visited" regions
    mask_flood[1:h+1, 1:w+1] = edge_barrier

    # Set barrier locations to 1 to "block"
    # FloodFill can only expand into 0 values
    mask_flood[mask_flood > 0] = 1

    # FloodFill from each seed point
    for seed in seeds:
        try:
            # Use a copy of the original image
            img_copy = image.copy()
            cv2.floodFill(img_copy, mask_flood, seed, (255, 255, 255),
                          tolerance, tolerance,
                          flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
        except:
            pass

    # Extract only the filled region from mask (where value is 255)
    result_mask = (mask_flood[1:h+1, 1:w+1] == 255).astype(np.uint8) * 255

    return result_mask, edge_barrier

def normalize_polygon(poly, num_points):
    """Normalize distorted polygon (straighten)"""
    if num_points == 4:
        # Quadrilateral -> correct to minimum area rectangle
        rect = cv2.minAreaRect(poly)
        box = cv2.boxPoints(rect)
        return np.int32(box)
    else:
        # Use original for pentagons and hexagons
        return poly

def fit_priority_polygon(contour):
    """Polygon priority: hexagon -> pentagon -> quadrilateral (with normalization)"""
    # First, flatten contour with convex hull (prevent inward distortion)
    hull = cv2.convexHull(contour)

    peri = cv2.arcLength(hull, True)
    found_polys = {}

    # Finer epsilon range (adjusted to 0.003~0.04)
    for eps_rate in np.linspace(0.003, 0.04, 150):
        epsilon = eps_rate * peri
        approx = cv2.approxPolyDP(hull, epsilon, True)
        pts = len(approx)
        if pts in [4, 5, 6]:
            if pts not in found_polys:
                found_polys[pts] = approx

    # Priority order: hexagon first
    if 6 in found_polys:
        return normalize_polygon(found_polys[6], 6), 6
    if 5 in found_polys:
        return normalize_polygon(found_polys[5], 5), 5
    if 4 in found_polys:
        return normalize_polygon(found_polys[4], 4), 4

    # If not found, use minimum area rectangle
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return np.int32(box), 4

def process_image_algorithm(image, img_name):
    h, w = image.shape[:2]

    # 1. Corner inspection
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cy, cx = h // 2, w // 2
    center_roi = lab_image[cy-10:cy+10, cx-10:cx+10]
    center_mean_lab = cv2.mean(center_roi)[:3]

    corners_map = {
        "TL": (0, 0), "TR": (0, w-CORNER_BOX_SIZE),
        "BL": (h-CORNER_BOX_SIZE, 0), "BR": (h-CORNER_BOX_SIZE, w-CORNER_BOX_SIZE)
    }
    object_seeds = [(w//2, h//2)]
    matched_corners_count = 0

    for name, (py, px) in corners_map.items():
        roi = lab_image[py:py+CORNER_BOX_SIZE, px:px+CORNER_BOX_SIZE]
        corner_mean = cv2.mean(roi)[:3]
        dist = calculate_color_distance(center_mean_lab, corner_mean)
        if dist < CORNER_DIFF_THRESH:
            object_seeds.append((px + CORNER_BOX_SIZE//2, py + CORNER_BOX_SIZE//2))
            matched_corners_count += 1

    # 2. Find object region using FloodFill (with barrier)
    mask_binary, edge_barrier = get_floodfill_mask(image, object_seeds, COLOR_TOLERANCE)

    # 2-1. Full Frame validation (when 4 corners match)
    if matched_corners_count == 4:
        # Check if there are barriers near corners
        corner_check_size = 30  # Size of region to check at corners
        corners_have_barrier = 0

        corner_regions = [
            edge_barrier[0:corner_check_size, 0:corner_check_size],  # TL
            edge_barrier[0:corner_check_size, w-corner_check_size:w],  # TR
            edge_barrier[h-corner_check_size:h, 0:corner_check_size],  # BL
            edge_barrier[h-corner_check_size:h, w-corner_check_size:w]  # BR
        ]

        for region in corner_regions:
            barrier_pixels = np.sum(region > 0)
            # If more than 5% of corner region is barrier, mark as "has barrier"
            if barrier_pixels > (corner_check_size * corner_check_size * 0.05):
                corners_have_barrier += 1

        # If all 4 corners have almost no barrier, it's a true Full Frame
        if corners_have_barrier == 0:
            print(f"  -> {img_name}: Full Frame Detected (Pass) - No barriers at corners")
            return image, None, "Full", None
        else:
            # Corners have barriers = object boundaries exist
            print(f"  -> {img_name}: False Full ({corners_have_barrier}/4 corners have barriers) - Processing as object")

    # 3. Post-processing: Fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # 4. Extract contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None, "Fail", mask_binary

    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Fit polygon (connect contour vertices)
    final_poly, pts_count = fit_priority_polygon(largest_contour)

    # 6. Generate final result using polygon mask
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(final_mask, [final_poly], 255)
    result = cv2.bitwise_and(image, image, mask=final_mask)

    return result, final_poly, pts_count, edge_barrier

# ==========================================
# Main execution
# ==========================================
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg'))]
print(f"📊 Processing {len(image_files)} images (Optimized Barrier - Canny {CANNY_THRESH_1}/{CANNY_THRESH_2})...")

for idx, img_name in enumerate(image_files):
    img_path = os.path.join(IMAGES_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None: continue

    result, poly, status, edges = process_image_algorithm(image, img_name)

    cv2.imwrite(os.path.join(PROCESSED_DIR, f"clean_{img_name.split('.')[0]}.png"), result)

    if idx % 5 == 0:
        debug = image.copy()

        # Display barrier in red
        if edges is not None:
            debug[edges > 0] = (0, 0, 255)

        if status == "Full":
            cv2.rectangle(debug, (0,0), (image.shape[1], image.shape[0]), (255, 0, 0), 10)
            cv2.putText(debug, "PASS (Full)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3)
        elif poly is not None:
            cv2.drawContours(debug, [poly], -1, (0, 255, 0), 5)
            # Draw vertices (larger and clearer)
            for i, pt in enumerate(poly):
                if len(pt.shape) == 2:  # [[x, y]] format
                    x, y = int(pt[0][0]), int(pt[0][1])
                else:  # [x, y] format
                    x, y = int(pt[0]), int(pt[1])

                # Large red circle
                cv2.circle(debug, (x, y), 20, (0, 0, 255), -1)
                # White border
                cv2.circle(debug, (x, y), 22, (255, 255, 255), 3)
                # Vertex number
                cv2.putText(debug, str(i+1), (x-10, y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.putText(debug, f"Poly: {status}-gon ({len(poly)} vertices)",
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        elif status == "Fail":
             cv2.putText(debug, "FAIL", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        cv2.imwrite(os.path.join(DEBUG_DIR, f"debug_barrier_{img_name}"), debug)
        print(f"✅ {img_name} Complete -> {status}")

print(f"\n🎉 Complete!")
print(f"📁 {DEBUG_DIR}:")
print(f"  - Red: Barrier (Canny {CANNY_THRESH_1}/{CANNY_THRESH_2})")
print(f"  - Green thick line: Final polygon")
print(f"  - Red large dots: Polygon vertices")
