import numpy as np
import cv2 as cv
import sys
import pyttsx3

def construct_yolo_v3():
    f = open('coco.names', 'r')
    class_names = [line.strip() for line in f.readlines()]
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names


def yolo_detect(img, yolo_model, out_layers):
    height, width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True)
    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    box, conf, id = [], [], []  # 박스, 신뢰도, 부류 정보를 저장할 리스트 생성
    for output in output3: # 세 개의 텐서를 각각 반복 처리
        for vec85 in output: # 85차원 백터(x,y,w,h,o,p1,p2,,,,,p80)를 반복 처리
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만 취함
                centerx, centery = int(vec85[0] * width), int(vec85[1] * height)
                w, h = int(vec85[2] * width), int(vec85[3] * height)
                x, y = int(centerx - w / 2), int(centery - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return objects


model, out_layers, class_names = construct_yolo_v3()  # YOLO 모델 생성
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔

cap = cv.VideoCapture(1)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
    engine = pyttsx3.init()
    res = yolo_detect(frame, model, out_layers)

    for i in range(len(res)):
        x1, y1, x2, y2, confidence, id = res[i]
        text1=str(class_names[id])
        text = str(class_names[id]) + '%.3f' % confidence
        cv.rectangle(frame, (x1, y1), (x2, y2), colors[id], 2)
        cv.putText(frame, text, (x1, y1 + 30), cv.FONT_HERSHEY_PLAIN, 1.5, colors[id], 2)
        # if text1=="coin" or text1=="pen" or text1=="eraser" or text1=="usb":
        #     engine.setProperty('voice', 200)
        #     engine.say(text1+"을 찾았습니다")
        #     engine.runAndWait()
    cv.imshow("Object detection from video by YOLO v.3", frame)

    key = cv.waitKeyEx(5) & 0xFF  # 키보드 입력받기

    if key == 27:  # ESC를 눌렀을 경우

        break  # 반복문 종료

cv.destroyAllWindows()  # 영상 창 닫기
cap.release()