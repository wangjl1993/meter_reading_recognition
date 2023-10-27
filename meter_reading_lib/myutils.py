
import cv2
import numpy as np
from typing import Union
from scipy.spatial.distance import cdist


def compute_iou_dist(boxA, boxB):
    # boxA = [int(x) for x in boxA]
    # boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return 1-iou


def match_by_iou(boxes: np.ndarray, match_boxes: np.ndarray):
    boxes, match_boxes = boxes.astype(np.int16), match_boxes.astype(np.int16)
    dists = cdist(boxes, match_boxes, compute_iou_dist)
    return dists.argmin(1)


def compute_degree_from_pt0_to_pt1(pt0, pt1):
    pt0, pt1 = pt0.reshape((1,-1)), pt1.reshape((1,-1))

    # 直线刚好接近垂直，直接返回 90°/270°
    if abs(pt0[0,0]-pt1[0,0]) <= 3:
        return 90 if pt0[0,1]< pt1[0,1] else 270
    
    pts = np.concatenate([pt0, pt1])
    k, _ = np.polyfit(pts[:, 0], pts[:, 1], 1)
    degree = convert_slope_to_degree360(k, pt1, pt0)
    degree = convert_image_coordinate_to_standard(degree)
    return degree

def a_is_close_to_b(a, b, thres):
    """判断a是否很靠近b，二者值相差在是否在thres内"""
    dist = abs(a-b)
    return True if dist<=thres else False


def generate_mask_from_line(k, b, expand_size, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    # 计算直线的起始和结束点坐标
    x_start = 0
    y_start = int(k * x_start + b)
    x_end = W - 1
    y_end = int(k * x_end + b)

    # 计算直线的单位向量
    line_direction = np.array([x_end - x_start, y_end - y_start], dtype=np.float32)
    line_direction /= np.linalg.norm(line_direction, ord=2)

    # 计算垂直于直线的单位向量
    perpendicular_direction = np.array([-line_direction[1], line_direction[0]])

    # 计算生成mask的四个顶点坐标
    offset = int(expand_size / 2)
    pt1 = np.array([x_start, y_start]) + offset * perpendicular_direction
    pt2 = np.array([x_start, y_start]) - offset * perpendicular_direction
    pt3 = np.array([x_end, y_end]) - offset * perpendicular_direction
    pt4 = np.array([x_end, y_end]) + offset * perpendicular_direction

    # 绘制多边形
    polygon_points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], color=1)
    return mask


def get_yolo_output_center(xyxy):
    ctr_x, ctr_y = (xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2
    ctr = np.array([[ctr_x, ctr_y]])
    return ctr


def mask_iou(mask1, mask2): 
    area1 = mask1.sum() 
    area2 = mask2.sum() 
    inter = ((mask1+mask2)==2).sum()
    iou = inter / (area1+area2-inter)
    return iou


def match_a_to_b(a: np.ndarray, b: np.ndarray):
    dists = cdist(a, b, 'euclidean')
    return dists.argmin(1)


# EPS = 1e-6
# OK = 0   # 正常
# NG = 1   # 异常
# UN = -1  # 未判断

def polygon2mask(points, fill_value, H, W, offset=None):

    mask = np.zeros((H, W), np.uint8)
    points = np.array(points, np.int32)
    if offset is not None:
        points = points - offset
    cv2.drawContours(mask,[points], -1, (fill_value), -1)
    return mask


def slope2degree(slope: Union[np.ndarray, float]):
    angle_rad = np.arctan(slope)      # 计算夹角弧度
    angle_deg = np.rad2deg(angle_rad) # 将弧度转换为角度 [-90, 90]
    return angle_deg


def convert_slope_to_degree360(k, pointer_end, meter_center):
    """降斜率转[0,360]角度, x轴正方向0°， y负方向90°，逆时针旋转。"""
    degree = slope2degree(k)
    if pointer_end[0, 0] > meter_center[0, 0]:
        degree = (-1*degree + 360) % 360
    else:
        degree = -1*degree + 180
    return degree


def convert_image_coordinate_to_standard(degree):
    """将图像坐标系（左上角原点）度数转标准坐标（左下角原点）系度数 """
    return 360-degree


def convert_image_coordinate_to_other(degree, offset):
    return (degree-offset+360) % 360

def fit_ellipse_from_mask(mask, min_num):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ells = []
    for contour in contours:
        if contour.shape[0] > min_num:
            ctr, axis_len, angle = cv2.fitEllipse(contour)    
            # angle_deg = compute_angle(ctr, contour)
            # axis_half_len = (axis_len[0]/2, axis_len[1]/2)
            ells.append((ctr, axis_len, angle)) 
    return ells


def fit_line_from_mask(mask):
    y, x = np.where(mask==True)
    k, b = np.polyfit(x, y, 1)
    return k, b


def fit_line_from_points(points):
    x, y = points[:, 0], points[:, 1]
    k, b = np.polyfit(x, y, 1)
    return k, b

def crop_img(img, xyxy):
    return img[xyxy[1]: xyxy[3], xyxy[0]:xyxy[2], :]

