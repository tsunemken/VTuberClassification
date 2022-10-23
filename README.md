# Abstract
画像分類の学習用にVtuber認識の画像分類するためのサンプルソースと学習済みモデルを公開しています。

勉強用なのでkerasのモデルを色々と試しています。

VTuver認識と言っていますが、配布予定モデルを使用しなければ何にでも使えると思います。

# 動作確認環境
## 環境
- OS：Ubuntu 20.04
- CPU：Intel Core i7-12700H
- メモリ：16GB
- GPU：GeForce RTX 3060 6GB

## CUDA関連
- CUDA：11.2
- cuDNN：8.6

## Python
- Python：3.8.10

## ライブラリ
- TensorFlow：2.7.0

あとは必要に応じて！

なお、dockerイメージ「tensorflow/tensorflow:2.7.0-gpu」を使用して環境を作成しています。

# データセットについて
データセットは非公開です（訓練済みのモデルは公開予定）。

データセットは配信からバストアップのスクリーンショットを撮って使用しています。

# クラス
2022年10月22日現在
```
amane-kanata
ars-almal
asumi-sena
fumino-tamaki
gawr-gura
hayama-marin
hoshikawa-sara
inugami-korone
minato-aqua
nakiri-ayame
nekomata-okayu
nishizono-chigusa
sakura-miko
sasaki-saku
shiina-yuika
shirakami-fubuki
shirayuki-mishiro
sorahoshi-kirame
tachibana-hinano
tokoyami-towa
usada-pekora
yorumi-rena
yukihana-lamy
```

# ディレクトリ構成
```
VTuberClassification/
├── README.md
├── src
│   ├── detect.py
│   ├── mlapp.py
│   ├── train.py
│   ├── utility.py
└── workdirectory
    ├── dataset
    ├── image
    │   ├── amane-kanata
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   ├── ars-almal
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   ├── asumi-sena
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   ├── ...
    ├── model
    │   └── DenseNet121.h5
    │   └── DenseNet169.h5
    │   └── DenseNet201.h5
    │   └── ...
    │   └── classes.txt
    └── result
        └── result.json
```

# 実行方法
- 学習
```
python train.py --model ResNet152V2 --img_size 224 --epochs 100 --batch_size 8 --learning_rate 0.001
```

- 推定
```
python detect.py --model ResNet152V2 --img_size 224 --source img/1.png
```

# License
VTuberClassification in under MIT License