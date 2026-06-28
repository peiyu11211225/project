import numpy as np

def calculate_angle(a, b, c):
    """計算向量 BA 與 BC 之間的夾角，b 為頂點"""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

def get_full_body_angles(pts):
    """
    傳入平滑後的 33 點座標，計算全身主要關節角度
    pts: dict {index: (x, y)}
    """
    angles = {}
    
    # --- 上肢 ---
    angles['L_elbow'] = calculate_angle(pts[11], pts[13], pts[15]) # 左肩-左肘-左腕
    angles['R_elbow'] = calculate_angle(pts[12], pts[14], pts[16]) # 右肩-右肘-右腕
    angles['L_shoulder'] = calculate_angle(pts[13], pts[11], pts[23]) # 左肘-左肩-左髖
    angles['R_shoulder'] = calculate_angle(pts[14], pts[12], pts[24]) # 右肘-右肩-右髖
    
    # --- 下肢 ---
    angles['L_knee'] = calculate_angle(pts[23], pts[25], pts[27]) # 左髖-左膝-左踝
    angles['R_knee'] = calculate_angle(pts[24], pts[26], pts[28]) # 右髖-右膝-右踝
    angles['L_hip'] = calculate_angle(pts[11], pts[23], pts[25])  # 左肩-左髖-左膝
    angles['R_hip'] = calculate_angle(pts[12], pts[24], pts[26])  # 右肩-右髖-右膝
    
    return angles