from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np
from typing import List, Union
from meter_reading_lib.myutils import (
    convert_image_coordinate_to_standard, 
    slope2degree, convert_image_coordinate_to_other,
    compute_degree_from_pt0_to_pt1
)



@dataclass
class DetOut:
    xyxy: np.ndarray
    xyxyn: np.ndarray
    xywh: np.ndarray
    xywhn: np.ndarray
    class_id: int
    conf: float
    label: str

    def __post_init__(self):
        self.xyxy = self.xyxy.astype(np.int16)
        self.xywh = self.xywh.astype(np.int16)
        self.center = self.xyxy[:2]



@dataclass
class SegOut(DetOut):
    mask: np.ndarray
    segments: np.ndarray



@dataclass
class PoseOut(DetOut):
    kpt: np.ndarray


@dataclass
class Pointer:

    pose_out: PoseOut
    reading: Union[float, int] = None
    is_abnormal: bool = False
    scale_coord_degree: float = None

    def __post_init__(self):
        self.tail, self.head = self.pose_out.kpt[0], self.pose_out.kpt[1]
        self.image_coord_degree = compute_degree_from_pt0_to_pt1(self.tail, self.head)

    def get_scale_coord_degree(self, offset):
        self.scale_coord_degree = convert_image_coordinate_to_other(self.image_coord_degree, offset)



@dataclass
class Scale:

    seg_out: SegOut
    image_coord_degree_range: List[List[float]] = field(default_factory=list)
    scale_coord_degree_range: List[List[float]] = field(default_factory=list) 
    

    def get_image_coord_degree_range(self, ctr):
        ctr_x, ctr_y = ctr
        denominator = self.seg_out.segments[:, 0]-ctr_x
        numerator = self.seg_out.segments[:, 1]-ctr_y
        slope = numerator / (denominator + 1e-6*(denominator==0))   # 避免分母为0，加1e-6
        
        # 计算每个轮廓点和指针起点组成直线的角度 [-90, 90]
        degrees = slope2degree(slope)

        # [-90, 90] --> [0, 360], 原点往X轴正方向为0度, 往Y轴负方向为90度, 往X轴负方向为180度, 顺时针旋转
        mask = (self.seg_out.segments[:, 0] > ctr_x) | ((self.seg_out.segments[:,0]==ctr_x)&(numerator<0))
        degrees[mask] = (-1*degrees[mask]+360) % 360
        degrees[~mask] = -1*degrees[~mask] + 180

        degrees = convert_image_coordinate_to_standard(degrees)
        self.image_coord_degree_range = self.get_scale_start_end(degrees)
        return self.image_coord_degree_range
    

    def get_scale_coord_degree_range(self, image_to_scale_coord_offset):
        assert len(self.image_coord_degree_range) > 0, "please call func: get_image_coord_degree_range first."
        scale_coord_degree_range = [ 
            [convert_image_coordinate_to_other(start, image_to_scale_coord_offset), convert_image_coordinate_to_other(end,image_to_scale_coord_offset)] \
                for start, end in self.image_coord_degree_range 
        ]

        if len(scale_coord_degree_range) == 1:
            self.scale_coord_degree_range = scale_coord_degree_range
            return self.scale_coord_degree_range

        res = []
        start, end = scale_coord_degree_range[0]
        n = len(scale_coord_degree_range[1:]) - 1
        for i, (s, e) in enumerate(scale_coord_degree_range[1:]):
            if s == end:
                end = e
            else:
                res.append([start, end])
                start, end = s, e
            if i == n:
                res.append([start, end])
        self.scale_coord_degree_range = res
        
        return self.scale_coord_degree_range
        
        
    @staticmethod
    def bulid_scale_coord(image_coord_degree_ranges):
        image_to_scale_coord_offset = 360
        for start, _ in image_coord_degree_ranges:
            if start > 0:
                if start < image_to_scale_coord_offset:
                    image_to_scale_coord_offset = start
        return image_to_scale_coord_offset
            

    @staticmethod
    def get_scale_start_end(degrees):

        # 统计每10度区间数量
        bins = 36
        cnts, interval = np.histogram(degrees, bins, range=(0,360))
        start_degree_interval_idx, end_degree_interval_idx = [], []
        res = []
        find_start = True
        for i, num in enumerate(cnts):
            if find_start:
                if num > 0:
                    start_degree_interval_idx.append(i)
                    find_start = False
            else:
                # 刻度末端区间查找规则：往后20度区间均没有角度，认为该区间为刻度末端
                the_latter = 0
                if i+1 < len(cnts):
                    the_latter += cnts[i+1]
                if i+2 < len(cnts):
                    the_latter += cnts[i+2]
                if the_latter==0:
                    if num > 0:
                        end_degree_interval_idx.append(i)
                    else:
                        end_degree_interval_idx.append(i-1)
                    find_start = True

        # 角度范围可能跨越0/360度，被分为2个区间。比如一个刻度表从Y轴正方向到Y负方向，则起止角度为 [0,90], [270,360]
        min_idx, max_idx = 0, len(cnts)-1
        for start_idx, end_idx in zip(start_degree_interval_idx, end_degree_interval_idx):
            if start_idx==min_idx and max_idx in end_degree_interval_idx:
                start_degree = 0
            else:
                start_degree_interval_mask = (degrees>=interval[start_idx])&(degrees<=interval[start_idx+1])
                start_degree = degrees[start_degree_interval_mask].min()
            if end_idx==max_idx and min_idx in start_degree_interval_idx:
                end_degree = 360
            else:
                end_degree_interval_mask = (degrees>=interval[end_idx])&(degrees<=interval[end_idx+1])
                end_degree = degrees[end_degree_interval_mask].max()
            res.append([start_degree, end_degree])

        return res


@dataclass
class Digit:

    pose_out: PoseOut
    ocr_out: Union[float, int] = None
    image_coord_degree: float = None
    scale_coord_degree: float = None
    
    def get_image_coord_degree(self, ctr: np.ndarray):
        self.image_coord_degree = compute_degree_from_pt0_to_pt1(ctr, self.pose_out.kpt)
    
    def get_scale_coord_degree(self, offset):
        assert self.image_coord_degree is not None, "Please call func: get_image_coord_degree first."
        self.scale_coord_degree = convert_image_coordinate_to_other(self.image_coord_degree, offset)

    @classmethod
    def build_from_det_out(cls, det_output: DetOut):
        """计数器表记数字，没有关键点，把中心作为关键点"""
        return cls(PoseOut(
            det_output.xyxy, det_output.xyxyn, det_output.xywh, det_output.xywhn,
            det_output.class_id, det_output.conf, det_output.label, det_output.center
        ))
    


@dataclass
class Meter:
    base_info: DetOut
    pointers: List[Pointer] = field(default_factory=list)
    scales: List[Scale] = field(default_factory=list)
    digits: List[Digit] = field(default_factory=list)
    reading: List = field(default_factory=list)
    msg: str = ""
    is_abnormal: bool = False

    def __post_init__(self):
        self.meter_type = self.base_info.label

    def collect_info(self):
        for pointer in self.pointers:
            self.reading.append(pointer.reading)
            self.is_abnormal = bool(self.is_abnormal+pointer.is_abnormal)

@dataclass
class Result:
    src_img: np.ndarray
    meters: List[Meter] = field(default_factory=list)
    is_abnormal: bool = False



    def plot(self):
        pass

    def get_info(self):
        msg = ""
        if len(self.meters) == 0:
            msg += "No meter found.\n"
        else:
            reading = []
            for meter in self.meters:
                reading += meter.reading
                msg += meter.msg
                self.is_abnormal = bool(self.is_abnormal + meter.is_abnormal)

        return self.is_abnormal, reading, msg