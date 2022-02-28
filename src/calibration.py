# 参考 ： https://qiita.com/ReoNagai/items/5da95dea149c66ddbbdd
#!/usr/bin/env python
# -*- coding: utf-8 -*
import numpy as np
import cv2
from cv2 import aruco
from harvesters.core import Harvester
import serial
import time
import math
import multiprocessing
import ctypes
import collections
import pickle
from PIL import Image
import matplotlib.pyplot as plt

camera_width = 1280
camera_height = 720

square_size = 1.85      # 正方形の1辺のサイズ[cm]
pattern_size = (7, 7)  # 交差ポイントの数
reference_img = 50 # 参照画像の枚数

# 画像表示の有無
capture_image = True

aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# カメラから取得したcomponentを画像に変換する
def capture_process(img_queue, capture_count):
    # カメラの設定
    h = Harvester()
    h.add_file('/Library/Frameworks/SentechSDK.framework/Libraries/libstgentl.cti')
    h.update()
    ia = h.create_image_acquirer(0)
    ia.remote_device.node_map.UserSetSelector.value = 'Default'
    ia.remote_device.node_map.ExposureAuto = 'Off'
    ia.remote_device.node_map.ExposureMode = 'Timed'
    ia.remote_device.node_map.TriggerSelector.value = 'FrameStart'
    ia.remote_device.node_map.TriggerMode.value = 'On'
    ia.remote_device.node_map.TriggerSource.value = 'Line0'
    ia.remote_device.node_map.Width.value = camera_width
    ia.remote_device.node_map.Height.value = camera_height
    # デジタルゲイン、ホワイトバランスゲイン(カラーのみ)
    ia.remote_device.node_map.Gain.value = 230.0
    print(ia.remote_device.node_map.PixelFormat.value)
    ia.start_acquisition()
    
    while True:
        with ia.fetch_buffer() as buffer:            
            component = buffer.payload.components[0].data
            image = component.reshape(camera_height, camera_width)
            img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
            # ctypes配列に変換
            np.ctypeslib.as_array(img_queue.get_obj())[:] = img
            capture_count.value += 1

def main():
    # マルチプロセス数 (macbook airだと4)
    process_num = 1
    #process_num = multiprocessing.cpu_count()
    print("process_num: ", process_num)

    # 処理した画像データを入れるキュー (ctypes配列)
    img_queue = multiprocessing.Value(
        ((ctypes.c_uint8 * 3) * camera_width) * camera_height)
    move_pointer_queue = multiprocessing.Value('i', 0)
    # 処理した画像の枚数を数える
    capture_count = multiprocessing.Value('i', 0)
    calc_count = multiprocessing.Value('i', 0)

    # プロセス生成
    process_set = []
    # 撮影プロセス
    process_ = multiprocessing.Process(
        target=capture_process, 
        args=(img_queue, capture_count))
    process_.start()
    process_set.append(process_)

    main_count = 0
    start_time = time.time()

    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    objpoints = []
    imgpoints = []

    while len(objpoints) < reference_img:
    
        main_count += 1
        # ctypes配列をnumpy配列に変換
        img = np.ctypeslib.as_array(img_queue.get_obj())

        if capture_image:
            cv2.imshow('drawDetectedMarkers', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェスボードが映らないとここで止まってしまう(なぜなのか...)
        ret, corner = cv2.findChessboardCorners(gray, pattern_size)
        
        # チェスボードが写っていれば
        if ret == True:
            print("detected coner!")
            print(str(len(objpoints)+1) + "/" + str(reference_img))
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(gray, corner, (5,5), (-1,-1), term)
            imgpoints.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加
            objpoints.append(pattern_points)

            time.sleep(0.2)        

    # 内部パラメータを計算
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 計算結果を保存
    np.save("mtx", mtx) # カメラ行列
    np.save("dist", dist.ravel()) # 歪みパラメータ
    # 計算結果を表示
    print("RMS = ", ret)
    print("mtx = \n", mtx)
    print("dist = ", dist.ravel())
    

if __name__ == '__main__':
    main()
