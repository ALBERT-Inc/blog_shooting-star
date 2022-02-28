import serial
import time
import math
import multiprocessing
import ctypes
import collections
import pickle
import cv2
from cv2 import aruco
import numpy as np
from harvesters.core import Harvester

from util import step2order

# 取得したデータを保存する場所
step_record_name = "./data/step_record.pickle"
# 大きくするほどレーザーが大きく動く
step_mag = 1
# レーザーとマーカーの位置の差がこれ以上の場合は記録しない(pixel)
min_diff = 3
# カメラ撮影画像表示の場合はTrue(処理は遅くなる)
capture_image = False
calibration = True

camera_width = 1280
camera_height = 720
resize_x = 640
resize_y = 360
no_ar_max_cnt = 50
no_pointer_max_cnt = 50
# ARマーカーの1辺の長さ(m)
marker_length = 0.059

aruco = cv2.aruco  # arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
camera_matrix = np.load("./data/mtx.npy")
distortion_coeff = np.load("./data/dist.npy")
ser = serial.Serial('/dev/cu.usbmodem14101', 9600)


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
    ia.remote_device.node_map.Gain.value = 239.0
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


def calc_process(i, img_queue, calc_count, no_ar_cnt, no_pointer_cnt):
    all_step_x = 0
    all_step_y = 0
    # 記録内容：ARマーカー座標[x, y, z], step数[x, y]
    step_record = []

    while True:
        calc_count.value += 1
        img = np.ctypeslib.as_array(img_queue.get_obj())
        img_small = cv2.resize(img, (resize_x, resize_y))
        pointer = np.where(np.sum(img_small, axis=2) > 750)

        # ポインターが見つからない場合、初期位置に戻る
        if len(pointer[0]) == 0:
            no_pointer_cnt.value += 1
            if no_pointer_cnt.value >= no_pointer_max_cnt:
                print("no_Pointer_reset")
                order = step2order()
                ser.write(order)
                no_pointer_cnt.value = 0
                all_step_x = 0
                all_step_y = 0
            continue

        else:
            pointer_x = int(np.mean(pointer[1]))
            pointer_y = int(np.mean(pointer[0]))
            no_pointer_cnt.value = 0

        # ARマーカーの位置検出
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)

        if len(corners) > 0:
            no_ar_cnt.value = 0
            # 左上の点から時計回り (x, y)
            x_ = int(corners[0][0][0][0] * (resize_x / camera_width))
            y_ = int(corners[0][0][0][1] * (resize_y / camera_height))

            # ARマーカーとレーザー位置の差分
            diff = max(abs(x_ - pointer_x), abs(y_ - pointer_y))
            print("diff: ", diff)
            # ARマーカー座標とstep数を記録
            if diff < min_diff:
                # 実空間のx,y,z座標を計算
                # y座標が上下逆になっていることに注意
                rvec, tvec, each_corner = aruco.estimatePoseSingleMarkers(corners[0], marker_length,
                                                                camera_matrix, distortion_coeff)
                tvec = np.squeeze(tvec + each_corner[0])
                step_record.append([round(tvec[0], 4), -round(tvec[1], 4), round(tvec[2], 4),
                                    all_step_x, all_step_y])

            step_x = int((x_ - pointer_x) * step_mag)
            step_y = -int((y_ - pointer_y) * step_mag)

            all_step_x += step_x
            all_step_y += step_y
            
            order = step2order("move", step_x, step_y)
            ser.write(order)
            while True:
                line = ser.readline()
                if line.decode().split("\r")[0] == "d":
                    break

        else:
            no_ar_cnt.value += 1

            # ARマーカーが見つからない場合、初期位置に戻る
            if no_ar_cnt.value >= no_ar_max_cnt:
                print("no_AR_marker_reset")
                order = step2order()
                ser.write(order)
                no_ar_cnt.value = 0
                all_step_x = 0
                all_step_y = 0
                # 記録の保存
                f = open(step_record_name, 'wb')
                pickle.dump(step_record, f)


def main():
    # マルチプロセス数
    # macbook airだとmultiprocessing.cpu_count() = 4
    process_num = 2
    #process_num = multiprocessing.cpu_count()
    print("process_num: ", process_num)

    # 処理した画像データを入れるキュー (ctypes配列)
    img_queue = multiprocessing.Value(
        ((ctypes.c_uint8 * 3) * camera_width) * camera_height)
    # 処理した画像の枚数を数える
    capture_count = multiprocessing.Value('i', 0)
    calc_count = multiprocessing.Value('i', 0)
    # ARマーカーが検出できなかった回数を記録
    no_ar_cnt = multiprocessing.Value('i', 0)
    # ポインターが検出できなかった回数を記録
    no_pointer_cnt = multiprocessing.Value('i', 0)

    time.sleep(2)
    # 最初にキャリブレーションを行う
    calibration_ = calibration
    if calibration_:
        mark_x = 0
        mark_y = 0
        order = step2order("move", 0, 0)
        ser.write(order)
        while True:
            print("レーザーを中心に合わせてください。")
            print("input: 'x,y' or 'done'")
            val = input()

            if val == "done":
                order = step2order("remark", mark_x, mark_y)
                ser.write(order)
                order = step2order()
                ser.write(order)
                pre_step_y = 0
                pre_step_x = 0
                no_ar_cnt.value = 1
                calibration_ = False
                print("calibration complete.")
                break

            step_x = int(val.split(",")[0])
            step_y = int(val.split(",")[1])
            mark_x += step_x
            mark_y += step_y
            order = step2order("move", step_x, step_y)
            ser.write(order)
    
    # プロセス生成
    process_set = []
    # 撮影プロセス
    process_ = multiprocessing.Process(
        target=capture_process,
        args=(img_queue, capture_count))
    process_.start()
    process_set.append(process_)

    # 計算プロセス (ARマーカー検出＋指示用ポインター位置計算)
    for i in range(process_num - 1):
        process_ = multiprocessing.Process(
            target=calc_process,
            args=(i, img_queue, calc_count, no_ar_cnt, no_pointer_cnt))
        process_.start()
        process_set.append(process_)


    main_count = 0
    start_time = time.time()
    # Cntl + c でexceptへ
    try:
        while True:
            main_count += 1

            if capture_image:
                # ctypes配列をnumpy配列に変換
                img = np.ctypeslib.as_array(img_queue.get_obj())
                # ARマーカー検出&描画
                corners, ids, rejectedImgPoints = aruco.detectMarkers(
                    img, dictionary)
                aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 255))
                # 画像を描画
                cv2.imshow('drawDetectedMarkers', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # プログラム終了時
    except KeyboardInterrupt:
        for i in range(process_num):
            print(process_set[i])
            process_set[i].terminate()

        print('stop!')
        end_time = time.time()
        print("capture_FPS:", np.round(
            capture_count.value / (end_time - start_time)))
        print("calc_FPS:", np.round(calc_count.value / (end_time - start_time)))
        print("main_FPS:", np.round(main_count / (end_time - start_time)))
        print("go_home.")
        order = step2order()
        ser.write(order)


if __name__ == '__main__':
    main()
