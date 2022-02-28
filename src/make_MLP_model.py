import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np

from model import DestStep

# 2回目にレーザー光が反射する高さ(m)
output_height = 0.0192
# モーター1周のstep数
motor_step = 25600

epoch = 15000
batch = 50
# 事前captureデータの場所
step_record_name = "./data/step_record.pickle"
model_name = "./data/model.pth"

f = open(step_record_name, "rb")
step_coordinate_record = pickle.load(f)
print("事前captureデータ数：", len(step_coordinate_record))

# 学習データと評価データに分割
step_len_80 = int(len(step_coordinate_record) * 0.8)
train_record = step_coordinate_record[:step_len_80]
test_record = step_coordinate_record[step_len_80:]

model = DestStep(output_height, motor_step)
opt = torch.optim.Adam(params=model.parameters(), lr=3e-4)

for i in range(epoch):
    data_ = torch.Tensor(random.sample(train_record, batch))
    x = data_[:, :3]
    t = data_[:, 3:]

    y = model(x)
    loss = F.mse_loss(y, t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 1000 == 0:
        print("loss:", np.round(loss.item(), 3))

print("\n学習結果")
print("回転(Rotation)")
print(list(model.cam2laser_layer.parameters())[0])
print("平行移動(Translation)")
print(list(model.cam2laser_layer.parameters())[1])

data_ = torch.Tensor(random.sample(test_record, 10))
x = data_[:, :3]
t = data_[:, 3:]
y = model(x)

print("教師データ")
print(np.round(t.detach().numpy()))
print("モデルの出力")
print(np.round(y.detach().numpy()))

torch.save(model.state_dict(), model_name)
