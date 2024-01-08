# # AI-利用神經網絡製作一个識別手寫數字的程序
南華大學跨領域-人工智能期中報告 11123024楊佳宜 11123007 陳奕瑄 11118128 吳佳恩

目錄
1.紅綠燈偵測資料集
2、使用KerasCV YOLOv8 進行物體偵測
3.驗證影像的推理
4、使用經過訓練的KerasCV YOLOv8 模型進行視訊推理
5.總結與結論

交通燈偵測資料集

使用紅綠燈偵測資料集訓練KerasCV YOLOv8 模型，小型紅綠燈資料集(S2TLD)由Thinklab提供，在筆記本的下載連結中提供了圖像和註釋的集合。資料集包含4564 個圖像，註釋以XML 格式呈現，以下圖片清晰地描繪了收集圖片的不同場景

將要使用的資料集版本包含四個類別
red
yellow
green
off

使用KerasCV YOLOv8 進行物體偵測，從設定必要的庫開始：
```python
!pip install keras-cv==0.5.1
!pip install keras-core
```
在初始步驟中，設定環境以利用「KerasCV YOLOv8」的功能進行物件偵測。安裝keras-cv 和keras-core 可確保開始物件偵測時，所需的所有模組的可用性，維護正確的版本防止相容性問題非常重要。本教程，我們使用keras-cv 0.5.1 版本，以獲得YOLOv8的最佳效果。

軟體包和庫導入

導入所需的軟體包和庫
```python
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras_cv
import requests
import zipfile
 
from tqdm.auto import tqdm
from tensorflow import keras
from keras_cv import bounding_box
from keras_cv import visualization
```
在深入研究「KerasCV YOLOv8」用於目標偵測的核心功能之前，讓我們先透過導入必要的函式庫和模組來奠定基礎：
os：幫助與Python 運行的底層作業系統進行交互，用於目錄操作；
xml.etree .ElementTree (ET)：協助解析XML 文件，常用於標註物件位置的資料集；
tensorflow & keras：KerasCV YOLOv8 "是實現深度學習功能的基礎；
keras_cv：一個重要的資料庫，為我們的專案提供了利用YOLOv8 模型的工具；
requests：允許發送HTTP請求，這對於取得線上資料集或模型權重是必不可少的；
zipfile：方便提取壓縮文件，在處理壓縮資料集或模型文件時可能有用
tqdm：透過進度條改進程式碼，讓冗長的程式變得簡單易用
來自keras_cv的bounding_box和visualization:在使用KerasCV YOLOv8檢測物件後，這些對於處理邊界框操作和視覺化結果至關重要。
確保導入這些模組後，我們就可以有效率地進行剩餘的物件偵測流程了。

下載資料集

首先，從直接來源下載紅綠燈偵測資料集
```python
# Download dataset.
def download_file(url, save_name):
    if not os.path.exists(save_name):
        print(f"Downloading file")
        file = requests.get(url, stream=True)
        total_size = int(file.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True
        )
        with open(os.path.join(save_name), 'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')
         
download_file(
    'https://www.dropbox.com/scl/fi/suext2oyjxa0v4p78bj3o/S2TLD_720x1280.zip?rlkey=iequuynn54uib0uhsc7eqfci4&dl=1',
    'S2TLD_720x1280.zip'
)
获取数据集请联系peaeci122
```
解壓縮資料集
```python
# Unzip the data file
def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall("./")
            print("Extracted all")
    except:
        print("Invalid file")
 
unzip('S2TLD_720x1280.zip')
```
資料集將會擷取到S2TLD_720x1280 目錄中。

資料集和訓練參數

需要定義適當的資料集和訓練參數，其中包括用於訓練和驗證的資料集拆分、批次大小、學習率以及KerasCV YOLOv8 模型需要訓練的epoch數。
```python
SPLIT_RATIO = 0.2
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCH = 75
GLOBAL_CLIPNORM = 10.0
```
其中20% 的數據用於驗證，其餘數據用於訓練。考慮到用於訓練的模型和圖像大小，批量大小為8，學習率設為0.001，模型將訓練75 個epoch。

資料集準備

這是訓練深度學習模型最重要的一個方面，我們首先要定義類別名，並存取所有圖像和註解檔案。
```python
class_ids = [
    "red",
    "yellow",
    "green",
    "off",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))
 
# Path to images and annotations
path_images = "S2TLD_720x1280/images/"
path_annot = "S2TLD_720x1280/annotations/"
 
# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
    ]
)
 
# Get all JPEG image file paths in path_images and sort them
jpg_files = sorted(
    [
        os.path.join(path_images, file_name)
        for file_name in os.listdir(path_images)
        if file_name.endswith(".jpg")
    ]
)
```
class_mapping 字典提供了從數位ID 到對應類別名稱的簡單查詢，所有映像和註解檔案路徑分別儲存在xml_files 和jpg_files 中。

接下來是解析XML 註解文件，以儲存訓練所需的標籤和邊界框註解。
```python
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
 
    image_name = root.find("filename").text
    image_path = os.path.join(path_images, image_name)
 
    boxes = []
    classes = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        classes.append(cls)
 
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
 
    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids
 
 
image_paths = []
bbox = []
classes = []
for xml_file in tqdm(xml_files):
    image_path, boxes, class_ids = parse_annotation(xml_file)
    image_paths.append(image_path)
    bbox.append(boxes)
    classes.append(class_ids)
```
