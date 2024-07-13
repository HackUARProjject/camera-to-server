from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# SSD MobileNetの設定
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# クラス名の読み込み（SSD MobileNet用）
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    result = detect_objects(img)

    return jsonify(result)

def detect_objects(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    result = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x2, y2) = box.astype("int")
            label = str(classes[class_id])
            result.append({
                'label': label,
                'confidence': float(confidence),
                'bbox': [x, y, x2, y2]
            })

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
