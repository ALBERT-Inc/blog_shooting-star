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
import torch
import torch.nn as nn
from harvesters.core import Harvester

from model import DestStep
from util import step2order

model_name = "./data/model.pth"
# カメラ撮影画像表示の場合はTrue(処理は遅くなる)
capture_image = False
calibration = True

camera_width = 1280
camera_height = 720
resize_x = 640
resize_y = 360
no_ar_max_cnt = 60

# 2回目にレーザー光が反射する高さ(m)
output_height = 0.0192
# モーター1周のstep数
motor_step = 25600

ser = serial.Serial('/dev/cu.usbmodem14101', 9600)

marker_length = 0.059  # [m]
aruco = cv2.aruco  # arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
camera_matrix = np.load("./data/mtx.npy")
distortion_coeff = np.load("./data/dist.npy")


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


def calc_process(i, img_queue, calc_count,
                 ar_marker_pos_x, ar_marker_pos_y,
                 ar_marker_pos_z, no_ar_cnt, calibration_flag):
    pre_img = np.zeros((resize_y, resize_x, 3))
    while True:
        calc_count.value += 1
        img = np.ctypeslib.as_array(img_queue.get_obj())
        img_small = cv2.resize(img, (resize_x, resize_y))
        img_small = img_small.astype(np.int16)
        diff_img = np.clip((img_small - pre_img), 0, 255)
        #print("all", np.max(np.sum(diff_img, axis=2).flatten()))
        #print(np.max(diff_img[:, :,0].flatten()), np.max(diff_img[:, :,1].flatten()), np.max(diff_img[:, :,2].flatten()))
        pointer = np.where((np.sum(diff_img, axis=2) > 90) * (diff_img[:, :, 2] < 80))
        #pointer = np.where((np.sum(diff_img, axis=2) > 150))
        pre_img = img_small
        if (calibration_flag.value) or (len(pointer[0]) > 0):
            if calibration_flag.value == 1:
                corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
                if len(corners) == 0:
                    continue
                no_ar_cnt.value = 0
                # マーカーの半分のサイズ[pixel]
                m_size = ((corners[0][0][1][0] - corners[0][0][0][0])
                          + (corners[0][0][2][1] - corners[0][0][1][1])) / 4

            elif len(pointer[0]) > 0:
                no_ar_cnt.value = 0
                #print(time.time())
                pointer_x = np.mean(pointer[1]) * (camera_width / resize_x)
                pointer_y = np.mean(pointer[0]) * (camera_height / resize_y)
                corners = np.array([[[[pointer_x-m_size, pointer_y+m_size],
                                      [pointer_x+m_size, pointer_y+m_size],
                                      [pointer_x+m_size, pointer_y-m_size],
                                      [pointer_x-m_size, pointer_y-m_size]]]])
            
            rvec, tvec, each_corner = aruco.estimatePoseSingleMarkers(corners[0], marker_length,
                                                            camera_matrix, distortion_coeff)
            tvec = np.squeeze(tvec + each_corner[0])
            # Y軸方向だけ上下が逆になっている
            ar_marker_pos_x.value = round(tvec[0], 4)
            ar_marker_pos_y.value = -round(tvec[1], 4)
            ar_marker_pos_z.value = round(tvec[2], 4)
        else:
            no_ar_cnt.value += 1


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
    # ARマーカーの座標
    ar_marker_pos_x = multiprocessing.Value('f', -1)
    ar_marker_pos_y = multiprocessing.Value('f', -1)
    ar_marker_pos_z = multiprocessing.Value('f', -1)
    no_ar_cnt = multiprocessing.Value('i', 1)
    calibration_flag = multiprocessing.Value('i', 1)

    # プロセス生成
    process_set = []
    # 撮影プロセス
    process_ = multiprocessing.Process(
        target=capture_process,
        args=(img_queue, capture_count))
    process_.start()
    process_set.append(process_)

    # 計算プロセス (ARマーカー検出)
    for i in range(process_num - 1):
        process_ = multiprocessing.Process(
            target=calc_process,
            args=(i, img_queue, calc_count,
                  ar_marker_pos_x, ar_marker_pos_y,
                  ar_marker_pos_z, no_ar_cnt, calibration_flag))
        process_.start()
        process_set.append(process_)

    # MLP
    model = DestStep(output_height, motor_step)
    model.load_state_dict(torch.load(model_name))

    # Arduinoの初期設定を待つ
    time.sleep(2)

    main_count = 0
    # 命令前のモーターのstep数
    pre_step_x = 0
    pre_step_y = 0
    calibration_flag.value = calibration
    start_time = time.time()

    # Cntl + c でexceptへ
    try:
        while True:
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

            # モーターに命令を送る
            if no_ar_cnt.value == 0:
                main_count += 1
                dir_step = model(torch.Tensor([[ar_marker_pos_x.value,
                                               ar_marker_pos_y.value, ar_marker_pos_z.value]]))[0]
                dir_step_x = int(dir_step[0])
                dir_step_y = int(dir_step[1])
                step_y = dir_step_y - pre_step_y
                step_x = dir_step_x - pre_step_x

                #動かない場合、命令は送らない
                if step_x == 0 and step_y == 0:
                    continue

                pre_step_y = pre_step_y + step_y
                pre_step_x = pre_step_x + step_x

                order = step2order("move", step_x, step_y)
                ser.write(order)

                while True:
                    line = ser.readline()
                    if line.decode().split("\r")[0] == "d":
                        break

                # 最初にARマーカーを検出したときにキャリブレーションを行う
                if calibration_flag.value:
                    mark_x = 0
                    mark_y = 0
                    order = step2order("move", 0, 0)
                    ser.write(order)
                    while True:
                        print("レーザーをARマーカーに合わせてください。")
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
                            calibration_flag.value = 0
                            print("calibration complete.")
                            break

                        step_x = int(val.split(",")[0])
                        step_y = int(val.split(",")[1])
                        mark_x += step_x
                        mark_y += step_y
                        order = step2order("move", step_x, step_y)
                        ser.write(order)

            else:
                if no_ar_cnt.value > no_ar_max_cnt:
                    print("no_ar_marker_reset")
                    order = step2order()
                    ser.write(order)
                    pre_step_y = 0
                    pre_step_x = 0
                    no_ar_cnt.value = 1

    # プログラム終了時処理
    except KeyboardInterrupt:
        for i in range(process_num):
            print(process_set[i])
            process_set[i].terminate()

        print('stop!')
        end_time = time.time()
        print("capture_FPS:", capture_count.value / (end_time - start_time))
        print("calc_FPS:", calc_count.value / (end_time - start_time))
        print("main_FPS:", main_count / (end_time - start_time))
        order = step2order()
        ser.write(order)


if __name__ == '__main__':
    main()
