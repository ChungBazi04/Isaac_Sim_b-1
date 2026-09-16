''' 
[EN]
Book Detect
A module that detects books and converts 
the coordinates of objects detected via camera (e.g., RealSense) 
into robot-compatible X, Y, Z spatial coordinates.

[KR]
책을 디텍팅하는 파일
realsense로 검출한객체의 좌표값을 로봇이 사용 가능한 x,y,z 좌표값으로 반환
'''


import cv2
import numpy as np
from ultralytics import YOLO

class BookDetector:
    def __init__(self, model_path='학습파일 경로 작성!!!!!'):
        # YOLO 모델 초기화
        self.model = YOLO(model_path)

    def process(self, rgb_image, depth_image, fx, fy, cx, cy):
        """
        RGB 이미지, Depth 이미지, 카메라 내부 파라미터를 받아
        책을 디텍팅하고 3D XYZ 좌표 리스트를 반환합니다.
        """
        # YOLO 추론 수행
        results = self.model(rgb_image, verbose=False)
        detected_targets = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                
                # 'book' 클래스인 경우에만 처리 (모델 설정에 맞게 이름 조정 가능)
                if self.model.names[cls_id] == 'book':
                    # 바운딩 박스 좌표 (xyxy)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 바운딩 박스 중심 픽셀(u, v) 계산
                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)

                    # 해당 픽셀의 Depth 값 추출
                    z = depth_image[v, u]
                    
                    if np.isnan(z) or z <= 0:
                        continue
                    if isinstance(z, np.uint16): # mm 단위인 경우 m 단위로 변환
                        z = float(z) / 1000.0
                    else:
                        z = float(z)

                    # 3D 역투영(Unprojection)으로 실제 공간 XYZ 좌표 계산
                    X = (u - cx) * z / fx
                    Y = (v - cy) * z / fy
                    Z = z

                    # 결과 저장 (좌표와 시각화용 박스 정보 함께 반환)
                    detected_targets.append({
                        'xyz': (X, Y, Z),
                        'box': (x1, y1, x2, y2),
                        'center': (u, v)
                    })

        return detected_targets