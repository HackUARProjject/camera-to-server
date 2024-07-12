import cv2  # OpenCVライブラリをインポート
import requests  # HTTPリクエストを送るためのライブラリをインポート

def capture_frame():
    cap = cv2.VideoCapture(0)  # カメラのIDが0の場合、カメラを開く
    if not cap.isOpened():
        print("カメラを開けません")  # カメラが開けなかった場合のエラーメッセージ
        return None  # フレームを取得できないのでNoneを返す

    ret, frame = cap.read()  # カメラからフレームを取得
    cap.release()  # カメラを解放

    if not ret:
        print("フレームを取得できませんでした")  # フレームが取得できなかった場合のエラーメッセージ
        return None  # フレームを取得できないのでNoneを返す

    return frame  # 取得したフレームを返す

def frame_to_jpeg(frame):
    ret, jpeg = cv2.imencode('.jpg', frame)  # フレームをJPEG形式にエンコード
    if not ret:
        print("フレームのエンコードに失敗しました")  # エンコードが失敗した場合のエラーメッセージ
        return None  # エンコードに失敗した場合はNoneを返す
    return jpeg.tobytes()  # エンコードされた画像データをバイト形式で返す

def send_frame_to_server(frame):
    url = "http://<サーバーのIPアドレス>:5000/detect"  # サーバーのURLを指定
    jpeg_data = frame_to_jpeg(frame)  # フレームをJPEG形式に変換
    if jpeg_data is None:
        return  # 変換に失敗した場合は処理を中止

    files = {'image': ('frame.jpg', jpeg_data, 'image/jpeg')}  # 送信するファイルを指定
    response = requests.post(url, files=files)  # POSTリクエストを送信

    if response.status_code == 200:
        print("サーバーからの応答:", response.json())  # サーバーからの応答を表示
    else:
        print("サーバーへのリクエストが失敗しました")  # リクエストが失敗した場合のエラーメッセージ

def main():
    frame = capture_frame()  # フレームをキャプチャ
    if frame is not None:
        send_frame_to_server(frame)  # フレームをサーバーに送信

if __name__ == "__main__":
    main()
