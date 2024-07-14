import cv2
import numpy as np
from datetime import datetime
import requests
import openai

# 画像を読み込む
image_path = "black.png"
image = cv2.imread(image_path)

# バウンディングボックスの色と太さを定義
bbox_color = (0, 255, 0)  
bbox_thickness = 40

# バウンディングボックスを描画
height, width, _ = image.shape
cv2.rectangle(image, (0, 0), (width, height - 50), bbox_color, bbox_thickness)  # 高さを少し減らして下にスペースを作る

# 現在の時刻を取得する
current_time = datetime.now().strftime('%H:%M')

# 時刻のテキストのフォント、サイズ、色を定義
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_color = bbox_color 
font_thickness = 5

# 画像に時刻のテキストを描画
cv2.putText(image, current_time, (30, 100), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

#右下のアクションボタンの表示
bottom_text = "アクションボタンを押して検索を開始"
bottom_font_scale = 1.5
bottom_font_thickness = 2

# テキストのサイズを取得
(text_width, text_height), baseline = cv2.getTextSize(bottom_text, font, bottom_font_scale, bottom_font_thickness)

# 右下のテキスト位置を計算
bottom_text_position = (width - text_width - 30, height - 20)

# 画像に下部のテキストを描画
cv2.putText(image, bottom_text, bottom_text_position, font, bottom_font_scale, font_color, bottom_font_thickness, cv2.LINE_AA)

# 画像を表示
cv2.imshow('Image with Bounding Box and Time', image)
cv2.waitKey(0)
cv2.destroyAllWindows()