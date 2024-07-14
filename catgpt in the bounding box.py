import openai
import cv2
import numpy as np
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# OpenAI APIキーの設定
openai.api_key = 'sk-proj-0xQyFvxZsCeINX3JrYomT3BlbkFJ9WJKUetKzZXmUW1Uyhhk'

# サンプルラベル名
label = "person"

# 説明文を取得する関数
def get_descriptions(label):
    descriptions = {}
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"オブジェクト '{label}' を日本語で20文字以内で詳細に説明してください。"}
            ],
            max_tokens=50
        )
        descriptions[label] = response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        descriptions[label] = "レート制限を超えました。しばらくしてからもう一度お試しください。"
    except openai.error.InvalidRequestError as e:
        descriptions[label] = f"無効なリクエスト: {e}"
    except Exception as e:
        descriptions[label] = f"エラーが発生しました: {e}"
    return descriptions

# ラベル名の説明文を取得
descriptions = get_descriptions(label)
description_text = descriptions.get(label, "")

#cv2が日本語対応ではないので、日本語にするための処理
def putText_japanese(img, text, point, size, color):
    #Raspberry Pi OSで実行する場合'/usr/share/fonts/opentype/NotoSansCJK-Bold.ttc'からフォントを指定してください
    #windows11でNotoSansCJK-BOLD.ttcを使用する場合こちらから"https://github.com/notofonts/noto-cjk/blob/main/Sans/OTC/NotoSansCJK-Bold.ttc"ダウンロードしてください(--;)
    font = ImageFont.truetype('C:/Users/tokag/AppData/Local/Microsoft/Windows/Fonts/NotoSansCJK-Bold.ttc', size)
    
    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)
    
    #drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)
    
    #テキスト描画
    draw.text(point, text, fill=color, font=font)
    
    #PILからndarrayに変換して返す
    return np.array(img_pil)
    
# 画像を読み込む
image_path = "black.png"
image = cv2.imread(image_path)

# バウンディングボックスの色と太さを定義
bbox_color = (0, 255, 0)  
bbox_thickness = 25

# バウンディングボックスを描画
height, width, _ = image.shape
cv2.rectangle(image, (0, 0), (width, height - 80), bbox_color, bbox_thickness)  # 高さを少し減らして下にスペースを作る

# 現在の時刻を取得する
current_time = datetime.now().strftime('%H:%M')

# 時刻のテキストのフォント、サイズ、色を定義
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_color = bbox_color 
font_thickness = 5

# 画像に時刻のテキストを描画
cv2.putText(image, current_time, (30, 100), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# 右下のアクションボタンの表示
bottom_text = "アクションボタンを押して検索を開始"

#アクションボタンの位置を調整(x座標, y座標)
bottom_text_position = (1400, image.shape[0] - 60) 

#説明文の位置を調整
description_position = (200, 600)

#日本語に対応させる
image = putText_japanese(image, bottom_text, bottom_text_position, 30, (0, 255, 0))
image = putText_japanese(image, description_text, description_position, 50, (0, 255, 0))

# ウィンドウサイズを設定して画像を表示
cv2.namedWindow('Image with Bounding Box, Time, and Description', cv2.WINDOW_NORMAL)
# ウィンドウサイズを800x600に設定
cv2.resizeWindow('Image with Bounding Box, Time, and Description', 800, 600)
cv2.imshow('Image with Bounding Box, Time, and Description', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
