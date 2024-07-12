from flask import Flask, request, jsonify  # Flask関連のモジュールをインポート
import cv2  # OpenCVライブラリをインポート
import numpy as np  # NumPyライブラリをインポート

app = Flask(__name__)  # Flaskアプリケーションのインスタンスを作成

# YOLOの設定
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # YOLOの重みと設定ファイルを読み込む
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]  # クラス名を読み込む
layer_names = net.getLayerNames()  # YOLOの全レイヤーの名前を取得
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # 出力レイヤーの名前を取得

@app.route('/detect', methods=['POST'])  # /detectエンドポイントを定義し、POSTメソッドでアクセス可能に
def detect():
    file = request.files['image']  # リクエストから画像ファイルを取得
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)  # 画像をデコードしてNumPy配列に変換

    result = detect_objects(img)  # 物体検出関数を呼び出して結果を取得

    return jsonify(result)  # 物体検出結果をJSON形式で返す

def detect_objects(img):
    height, width, channels = img.shape  # 画像の高さ、幅、チャンネル数を取得
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # 画像をYOLO用に前処理
    net.setInput(blob)  # 前処理した画像をYOLOモデルに入力
    outs = net.forward(output_layers)  # YOLOモデルで前方伝播し、出力を取得

    class_ids = []  # クラスIDを格納するリスト
    confidences = []  # 信頼度を格納するリスト
    boxes = []  # バウンディングボックスを格納するリスト

    for out in outs:
        for detection in out:
            scores = detection[5:]  # 各クラスのスコアを取得
            class_id = np.argmax(scores)  # 最も高いスコアのクラスIDを取得
            confidence = scores[class_id]  # 最も高いスコアを信頼度として取得
            if confidence > 0.5:  # 信頼度が0.5以上の場合
                center_x = int(detection[0] * width)  # バウンディングボックスの中心のx座標
                center_y = int(detection[1] * height)  # バウンディングボックスの中心のy座標
                w = int(detection[2] * width)  # バウンディングボックスの幅
                h = int(detection[3] * height)  # バウンディングボックスの高さ
                x = int(center_x - w / 2)  # バウンディングボックスの左上のx座標
                y = int(center_y - h / 2)  # バウンディングボックスの左上のy座標
                boxes.append([x, y, w, h])  # バウンディングボックスをリストに追加
                confidences.append(float(confidence))  # 信頼度をリストに追加
                class_ids.append(class_id)  # クラスIDをリストに追加

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # 非極大抑制を適用して重複するバウンディングボックスを削除

    result = []  # 結果を格納するリスト
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # クラスIDからラベルを取得
            result.append({
                'label': label,
                'confidence': confidences[i],
                'bbox': [x, y, x+w, y+h]
            })

    return result  # 物体検出の結果を返す

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # サーバーを0.0.0.0:5000で起動
