
import numpy as np
from meter_reading_lib.model_infer import (
    yolov8_det_infer, yolov8_seg_infer,
    yolov8_pose_infer, ocr_model_infer
)
from meter_reading_lib.myutils import (
    crop_img,
    a_is_close_to_b,
    match_by_iou,
    
)
import torch
import torch.nn as nn
from meter_reading_lib.dataStruct import (
    Pointer, Scale, Digit,
    Meter, Result
)

from typing import List


def is_pointer_degree_abnormal(degree, degree_ranges, thres=10):
    """指针读数异常：指针角度在刻度表角度范围外；指针角度在刻度表的起止位置。"""
    min_degree, max_degree = 0, 360

    # 刻度范围360度，不会超过刻度范围，返回正常
    span = sum([end-start for (start,end) in degree_ranges])
    if span == max_degree:
        return 0, "Scale_span=360.\n"
    
    
    cross_origin = False
    if min_degree in sum(degree_ranges, []) and max_degree in sum(degree_ranges, []):
        cross_origin = True   # 0/360 在刻度范围内， 刻度表读数范围经过了起止点，被分为2部分。

    msg = f"Degree_ranges={degree_ranges}, pointer_degree={degree}.\n"
    for start, end in degree_ranges:
        if start<degree<end:
            dist_to_start = degree-start
            dist_to_end = end-degree
            if cross_origin:
                if (start!=min_degree and dist_to_start<thres) or (end!=max_degree and dist_to_end<thres):
                    return 1, msg
                else:
                    return 0, msg
            else:
                # 角度距离刻度表起止角度很近，认为异常。 
                if dist_to_end<thres or dist_to_start<thres:
                    return 1, msg
                else:
                    return 0, msg
    return 1, msg


def compute_ratio_value(degree, d1, d2, v1, v2):
    degree_span = abs(d2-d1)
    value_span = abs(v2-v1)
    value = v1 + value_span*(degree-d1)/degree_span
    return value


def compute_non_counter_meter_reading(pointer: Pointer, digits: List[Digit]):
    selected_digits = sorted([d for d in digits if d.ocr_out is not None], key= lambda d: d.scale_coord_degree)
    if len(selected_digits) < 2:
        reading, msg = None, "Not enough ocr digits for recognition."
    else:
        n = len(selected_digits)
        for i, digit in enumerate(selected_digits):
            if digit.scale_coord_degree > pointer.scale_coord_degree:
                if i == 0:
                    d1, d2 = selected_digits[0].scale_coord_degree, selected_digits[1].scale_coord_degree
                    v1, v2 = selected_digits[0].ocr_out, selected_digits[1].ocr_out
                    break
                else:
                    d1, d2 = selected_digits[i-1].scale_coord_degree, selected_digits[i].scale_coord_degree
                    v1, v2 = selected_digits[i-1].ocr_out, selected_digits[i].ocr_out
                    break
            
            if i+1 == n:
                d1, d2 = selected_digits[i-1].scale_coord_degree, selected_digits[i].scale_coord_degree
                v1, v2 = selected_digits[i-1].ocr_out, selected_digits[i].ocr_out
        reading = compute_ratio_value(pointer.scale_coord_degree, d1, d2, v1, v2)
        msg = f"Non_counter_meter pointer_degree={pointer.scale_coord_degree} reading={reading:.2f}, compute reading according to d1={d1:.2f}, v1={v1:.2f}, d2={d2:.2f}, v2={v2:.2f}.\n"
    return reading, msg


def non_counter_meter_reading_recognition(
        meter: Meter, img: np.ndarray, ocr_model: nn.Module, 
        device: torch.device, thres: float=15
    ):
    if len(meter.pointers) == 0:
        meter.msg += "No pointer found.\n"
        return 
    if len(meter.scales) == 0:
        meter.msg += "No scale found.\n"
        return 
    
    for digit in meter.digits:
        digit_img = crop_img(img, digit.pose_out.xyxy)
        ocr_pre = ocr_model_infer(ocr_model, digit_img, device)
        digit.ocr_out = ocr_pre
    
    green_scales = [s for s in meter.scales if s.seg_out.label == "green_scale"]
    other_scales = [s for s in meter.scales if s.seg_out.label != "green_scale"]

    for pointer in meter.pointers:
        
        ############################################# 指针异常判断 #############################################
        # 所有刻度（绿色+非绿色）计算角度范围
        all_scale_degree_range = [s.get_image_coord_degree_range(pointer.tail) for s in meter.scales]
        all_scale_degree_range = sum(all_scale_degree_range, [])
        image_to_scale_coord_offset = Scale.bulid_scale_coord(all_scale_degree_range)
        _ = [s.get_scale_coord_degree_range(image_to_scale_coord_offset) for s in meter.scales ]
        # 判断指针指向是否异常，有绿色刻度优先考虑绿色刻度
        if len(green_scales) > 0:
            thres = 2
            normal_degree_range = sum([s.scale_coord_degree_range for s in green_scales], [])
        else:
            normal_degree_range = sum([s.scale_coord_degree_range for s in other_scales], [])
        
        pointer.get_scale_coord_degree(image_to_scale_coord_offset)
        is_pointer_abnormal, _ = is_pointer_degree_abnormal(pointer.scale_coord_degree, normal_degree_range, thres)
        pointer.is_abnormal = is_pointer_abnormal

        ############################################# 指针读数识别 #############################################
        
        # 指针是否在刻度范围内

        is_pointer_point_to_scale = False
        for start, end in all_scale_degree_range:
            if  start <= pointer.image_coord_degree <= end:
                is_pointer_point_to_scale = True
                break
        if is_pointer_point_to_scale:
            for digit in meter.digits:
                digit.get_image_coord_degree(pointer.tail)
                digit.get_scale_coord_degree(image_to_scale_coord_offset)
            
            reading, tmp_msg = compute_non_counter_meter_reading(pointer, meter.digits)
            pointer.reading = reading
            meter.msg += tmp_msg
        




def counter_meter_reading_recognition(meter: Meter, img: np.ndarray, ocr_model: nn.Module, device: torch.device):
    if len(meter.pointers) == 0:
        meter.is_abnormal = True
        meter.msg += "No pointer found.\n"
        return 
    
    if len(meter.digits) == 0:
        meter.msg += "No digit found.\n"
        return 
 
    for i, pointer in enumerate(meter.pointers):
    
        pointer.is_abnormal = True
        for digit in meter.digits:
            digit.get_image_coord_degree(pointer.tail)
            if a_is_close_to_b(pointer.image_coord_degree, digit.image_coord_degree, 15):
                pointer.is_abnormal = False

                digit_img = crop_img(img, digit.pose_out.xyxy)
                ocr_pre = ocr_model_infer(ocr_model, digit_img, device)
                if ocr_pre is not None:
                    digit.ocr_out = ocr_pre
                    pointer.reading = int(ocr_pre)
                    meter.msg += f"Counter_meter pointer{i} reading is {pointer.reading}\n"
                    return
                else:
                    meter.msg += f"Counter_meter ocr pre faild, attempt pre the opposite digit...\n"

        # 指针可能把数字遮挡，尝试求指针反方向的数字
        pointer_reversed_degree = (pointer.image_coord_degree+180)%360
        for digit in meter.digits:
            if a_is_close_to_b(pointer_reversed_degree, digit.image_coord_degree, 15):
                pointer.is_abnormal = False

                digit_img = crop_img(img, digit.pose_out.xyxy)
                ocr_pre = ocr_model_infer(ocr_model, digit_img, device)
                if ocr_pre is not None:
                    digit.ocr_out = ocr_pre
                    pointer.reading = (int(ocr_pre)+5)%10
                    meter.msg += f"Counter_meter pointer{i} reading is {pointer.reading}.\n"
                    return
                else:
                    meter.msg += f"Counter_meter ocr pre faild.\n"

    return




def process_meter(meters: List[Meter], img: np.ndarray, ocr_model: nn.Module, device: torch.device, thres:float=15):
    for i, meter in enumerate(meters):
        meter.msg += f"Process meter{i}, meter_type={meter.meter_type}.\n"
        if meter.meter_type == "counter_meter":
            counter_meter_reading_recognition(meter, img, ocr_model, device)
        else:
            non_counter_meter_reading_recognition(meter, img, ocr_model, device, thres)
        
        meter.collect_info()



def meter_reading_recognition(
        img, device,
        det_meter_digit_model, det_meter_digit_model_params,
        seg_scale_model,  seg_scale_model_params,
        scale_kpts_model, scale_kpts_model_params,
        pointer_kpts_model, pointer_kpts_model_params, ocr_model
    ):

    

    det_outputs = yolov8_det_infer(det_meter_digit_model, img, device=device, **det_meter_digit_model_params)  # 检测图中多少个表和 计数器数字
    meter_ouputs = [i for i in det_outputs if i.label in ["counter_meter", "non-counter_meter"]]               # 1: 非计数器表计，2：计数器表计
    
    scale_seg_outputs = yolov8_seg_infer(seg_scale_model, img, device=device, **seg_scale_model_params)              # 刻度表分割
    scale_kpt_outputs = yolov8_pose_infer(scale_kpts_model, img, device=device, **scale_kpts_model_params)
    pointer_kpts_outputs = yolov8_pose_infer(pointer_kpts_model, img, device=device, **pointer_kpts_model_params)     # 指针关键点


    digits = [Digit.build_from_det_out(i) for i in det_outputs if i.label=="digit"]    
    scales = [Scale(i) for i in scale_seg_outputs]
    scale_digits = [Digit(i) for i in scale_kpt_outputs]
    pointers = [Pointer(i) for i in pointer_kpts_outputs]


    meter_num = len(meter_ouputs)

    # 未检测到表
    if meter_num == 0:
        res = Result(img)
        return res

    # 图中有1个表
    elif meter_num == 1:
        meter_type = meter_ouputs[0].label
        meters = [Meter(meter_ouputs[0], pointers, scales, digits if meter_type=="counter_meter" else scale_digits)]
        
    # 图中有多个表，将每个表匹配上对应的刻度，指针，数字。再检测。
    else:

        meters = [Meter(i) for i in meter_ouputs]
        meters_box = np.stack([m.base_info.xyxy for m in meters])

        if len(pointers) > 0:
            pointers_box = np.stack([p.pose_out.xyxy for p in pointers])
            match_res = match_by_iou(pointers_box, meters_box)
            for p, idx in zip(pointers, match_res):
                meters[idx].pointers.append(p)

        if len(scales) > 0:
            scales_box = np.stack([s.seg_out.xyxy for s in scales])
            match_res = match_by_iou(scales_box, meters_box)
            for s, idx in zip(scales, match_res):
                meters[idx].scales.append(s)

        # 带数字的刻度关键点
        if len(scale_digits) > 0:
            scale_digits_box = np.stack([sd.pose_out.xyxy for sd in scale_digits])
            match_res = match_by_iou(scale_digits_box, meters_box)
            for scale_digit, idx in zip(scale_digits, match_res):
                if meters[idx].meter_type == "non-counter_meter":
                    meters[idx].digits.append(scale_digit)
        
        if len(digits) > 0:
            digits_box = np.stack([d.pose_out.xyxy for d in digits])
            match_res = match_by_iou(digits_box, meters_box)
            for d, idx in zip(digits, match_res):
                if meters[idx].meter_type == "counter_meter":
                    meters[idx].digits.append(d)


    process_meter(meters, img, ocr_model, device)
    res = Result(img, meters)
    return res




