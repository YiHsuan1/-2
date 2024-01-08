# # AI-利用神經網絡製作一个識別手寫數字的程序
南華大學跨領域-人工智能期中報告 11123024楊佳宜 11123007 陳奕瑄 11118128 吳佳恩

原文链接：

https://xiaoling.blog.csdn.net/article/details/133855492

https://blog.csdn.net/lxiao428/article/details/133877333

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
KerasCV YOLOv8模型

使用COCO 預訓練主幹創建KerasCV YOLOv8 模型，主幹是YOLOv8 Large，整個預訓練模型中，先用COCO 預訓練權重加載主幹網，然後使用隨機初始化的頭部權重創建整個YOLOv8 模型。
```python
backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_l_backbone_coco",
    load_weights=True
)
 
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=3,
)
 
yolo.summary()
```
設定load_weights = True 很重要，否則COCO 預訓練的權重將無法載入到主幹網路。

由於資料集註解檔案是XML 格式，所有的邊界框都是XYXY 格式，因此上述程式碼區塊中的bounding_box_format 為「xyxy」。此外，根據KerasCV YOLOv8 官方文檔，fpn_depth 為3。

下一步：定義最佳化器並編譯模型
optimizer = tf.keras.optimizers.Adam(
learning_rate=LEARNING_RATE,
global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
optimizer=optimizer, classification_loss=“binary_crossentropy”, box_loss=“ciou”
)

學習率按照先前的定義進行設置，梯度剪切則使用global_clipnorm 參數，這確保了影響模型參數更新的梯度不會變得太大而破壞訓練的穩定性。

優化器準備好後，繼續編譯YOLOv8 模型，這樣模型就可以使用下面定義的損失函數進行訓練了：

· Classification_loss：選擇「binary_crossentropy」作為分類損失；

· box_loss:「ciou」或「Complete Intersection over Union」是一種先進的邊界框損失函數，它可以解釋預測框和真實框之間的大小和形狀差異。

最終建立的模型包含4,100 萬個參數，以下是模型摘要的片段，以及可訓練參數的數量。

評估指標

我們選擇平均值(mAP) 作為評估指標，KerasCV 已經為他所有目標偵測模型提供了mAP 的最佳化實作。

```python
class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )
 
        self.save_path = save_path
        self.best_map = -1.0
 
    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)
 
        metrics = self.metrics.result(force=True)
        logs.update(metrics)
 
        current_map = metrics["MaP"]
        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)  # Save the model when mAP improves
 
        return logs
```
使用自訂Keras 回呼來定義EvaluateCOCOMetricsCallback，在每次驗證循環後執行，如果目前的mAP 大於先前的最佳mAP，那麼模型權重將會儲存到磁碟中。


Tensorboard回呼日誌

我們也要定義一個Tensorboard 回調，用於自動記錄所有mAP 和損失圖。
```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs_yolov8large")
```
所有Tensorboard 日誌都將儲存在logs_yolov8large 目錄中。
對KerasCV YOLO8模型進行紅綠燈偵測的訓練
現在可以開始訓練過程了，所有元件都已準備就緒，只需呼叫yolo.fit() 方法就可以開始訓練。
```python
history = yolo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH,
    callbacks=[
        EvaluateCOCOMetricsCallback(val_ds, "model_yolov8large.h5"),
        tensorboard_callback
    ],
)
```
train_ds 和val_ds 分別用作訓練資料集和驗證資料集，而且我們還提供了上一節中定義的回調，mAP 和loss 的值將儲存在歷史變數中。由於所有資料都會記錄到Tensorboard 中，所以並不需要這些資料。

YOLOv8 模型的最佳mAP 超過48%，這也是保存最佳模型權重的地方。

驗證影像的推理

由於我們現在已經有了訓練好的模型，因此可以用它來對驗證集的圖像進行推理。
```python
def visualize_detections(model, dataset, bounding_box_format):
    for i in range(10):
        images, y_true = next(iter(dataset.take(i+1)))
        y_pred = model.predict(images)
        y_pred = bounding_box.to_ragged(y_pred)
        visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            bounding_box_format=bounding_box_format,
            # y_true=y_true,
            y_pred=y_pred,
            scale=4,
            rows=2,
            cols=2,
            show=True,
            font_scale=0.7,
            class_mapping=class_mapping,
        )
visualize_detections(yolo, dataset=val_ds, bounding_box_format="xyxy")
```
上述函數對資料循環10 次進行推理，每次推理後，都會使用KerasCV 內建的plot_bounding_box_gallery 函數繪製結果。

下圖顯示了一些預測正確的結果
模型對所有紅綠燈的預測都是正確的。

儘管模型準確度非常高了，但還不夠完美，以下是一些預測結果不正確的圖片。
上圖顯示了一個圖像實例，模型將建築物的窗戶預測為紅綠燈，在另一個例子中，它缺少綠色和紅色紅綠燈的預測。

為了緩解上述情況，除了水平翻轉之外，還可以對影像進行更多增強。KerasCV 有許多增強功能，可用於減少過度擬合並提高不同情況下的準確性。

使用模型進行視訊推理

使用經過訓練的KerasCV YOLOv8 模型進行視頻推理，在可下載內容中找到視頻推理腳本，並在自己的視頻上運行推理，下面是運行視頻推理的示例命令。
```python
python infer_video.py --input inference_data/video.mov
```
輸入（–input）標記會取得運行推理的視訊檔案的路徑，下面是視訊推理實驗的輸出範例。

結果看起來不錯，幾乎在所有幀中，模型都能正確檢測到紅綠燈，當然也有一點閃爍，但很有可能在經過更多的訓練和增強後就會消失。
使用訓練有素的KerasCV YOLOv8 模型進行紅綠燈

總結與結論

本文到此結束，從KerasCV 的初始設定開始，然後進入紅綠燈偵測資料集，詳細介紹了YOLOv8 偵測模型的準備工作，隨後進行了訓練和驗證。
