# 流れ星に全自動で願いを送る装置を作る
ブログ記事、[流れ星に全自動で願いを送る装置を作る](https://blog.albert2005.co.jp/2022/02/28/shooting-star/)の実験で使用したプログラムです。

## 実行方法
- `requirements.txt` に書かれているライブラリをインストールしてください．
- その後、`src`に格納されている以下のプログラムを順番に実行してください。
    1. `calibration.py`
    2. `pre_capture.py`
    3. `make_MLP_model.py`
- ARマーカーの追跡実験をやる場合は`AR_marker_capture.py`を、流れ星の追跡実験をやる場合は`star_capture.py`を実行してください。