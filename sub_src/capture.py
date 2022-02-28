import multiprocessing
import ctypes
import cv2
import time
import numpy as np
from harvesters.core import Harvester

camera_width = 1280
camera_height = 720

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


def main():
    # マルチプロセス数
    process_num = 1
    print("process_num: ", process_num)

    # 処理した画像データを入れるキュー (ctypes配列)
    img_queue = multiprocessing.Value(
        ((ctypes.c_uint8 * 3) * camera_width) * camera_height)
    # 処理した画像の枚数を数える
    capture_count = multiprocessing.Value('i', 0)

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
    # Cntl + c でexceptへ
    try:
        while True:
            main_count += 1

            # ctypes配列をnumpy配列に変換
            img = np.ctypeslib.as_array(img_queue.get_obj())
            # 画像を描画
            cv2.imshow('capture', img)
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
        print("main_FPS:", np.round(main_count / (end_time - start_time)))


if __name__ == '__main__':
    main()