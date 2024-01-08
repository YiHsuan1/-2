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
#获取数据集请联系peaeci122
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
parse_annotation(xml_file) 函數會深入研究每個XML 文件，提取文件名稱、物件類別及其各自的邊界框座標。在class_mapping 的幫助下，它會將​​類別名稱轉換為類別ID，以便於使用。
解析完所有XML檔案後，我們在單獨的清單中收集所有影像路徑、邊界框和類別id，然後使用tf.data.Dataset.from_tensor_slices 將它們組合成TensorFlow資料集。
```python
bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)
 
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
```
所有資料並未儲存在單一tf.data.Dataset物件中，需要使用SPLIT_RATIO 將其分為訓練集和驗證集。
```python
# Determine the number of validation samples
num_val = int(len(xml_files) * SPLIT_RATIO)
 
# Split the dataset into train and validation sets
val_data = data.take(num_val)
train_data = data.skip(num_val)
```
載入圖像和註釋，並套用所需的預處理。
```python
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image
 
def load_dataset(image_path, classes, bbox):
    # Read Image
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": bbox,
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}
 
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640),
            scale_factor=(1.0, 1.0),
            bounding_box_format="xyxy",
        ),
    ]
)
 
train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 4)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
```
對於訓練集，將影像調整為640×640 分辨率，並應用隨機水平翻轉增強，增強將確保模型不會過早過度擬合。
至於驗證集，則不需要任何增強，只需調整影像大小就足夠了。
```python
resizing = keras_cv.layers.JitteredResize(
    target_size=(640, 640),
    scale_factor=(1.0, 1.0),
    bounding_box_format="xyxy",
)
 
val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.shuffle(BATCH_SIZE * 4)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
```
在進入下一階段之前，使用上面創建的訓練和驗證資料集來視覺化一些樣本。
```python
def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )
 
 
visualize_dataset(
    train_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)
 
visualize_dataset(
    val_ds, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2
)
```
以下是上述視覺化功能的一些輸出結果

最後，建立最終的資料集格式
```python
def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]
 
 
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
 
val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```
為方便模型訓練，使用dict_too_tuple 函數對資料集進行轉換，並透過預取功能對資料集進行最佳化，以獲得更好的效能。
