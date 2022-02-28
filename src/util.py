import torch

# step数を命令に変換
def step2order(choice="reset", step_x=0, step_y=0):
    # mark位置に戻る
    if choice == "reset":
        order = "reset"
    # mark位置を更新
    elif choice == "remark":
        order = "remark," + str(-step_x) + "," + str(-step_y)
    # home位置に戻る
    elif choice == "home":
        order = "home"
    # モーターを動かす
    # dir : 0: 時計回り、1: 反時計回り
    else:
        if step_x < 0:
            dir_1 = 1
            step_x = abs(step_x)
        else:
            dir_1 = 0
        if step_y > 0:
            dir_2 = 0
        else:
            dir_2 = 1
            step_y = abs(step_y)
        order = (str(dir_1) + "," + str(step_x) + ","
                 + str(dir_2) + "," + str(step_y))
        
    return str.encode(str("0") + order + "\n")


# 回転表現間の相互変換
def _euler2matrix_x(a):
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)

    matrix = torch.stack([
        torch.cat([o, z, z], axis=1),
        torch.cat([z, c, -s], axis=1),
        torch.cat([z, s, c], axis=1)
    ], axis=1)
    return matrix


def _euler2matrix_y(a):
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)

    matrix = torch.stack([
        torch.cat([c, z, s], axis=1),
        torch.cat([z, o, z], axis=1),
        torch.cat([-s, z, c], axis=1)
    ], axis=1)
    return matrix


def _euler2matrix_z(a):
    c = torch.cos(a)
    s = torch.sin(a)
    z = torch.zeros_like(a)
    o = torch.ones_like(a)

    matrix = torch.stack([
        torch.cat([c, -s, z], axis=1),
        torch.cat([s, c, z], axis=1),
        torch.cat([z, z, o], axis=1)
    ], axis=1)
    return matrix


def euler2matrix(x, euler):
    """Convert euler angles to rotatin matrix.

    Note:
        This function does not provide all features of
        ``scipy.spatial.transform.Rotation``.

    Args:
        euler (Tensor, [N, 3] or [3,]): Euler angles specified in radians.
            These three values are interpreted as angles of intrinsic
            rotations around the x, y, z axes.

    Returns:
        matrix (Tensorm [N, 3, 3] or [3, 3]): Rotation matrix.

    """
    stacked = True
    if euler.dim() == 1:
        euler = euler[None]

    matrix_x = _euler2matrix_x(euler[:, 0:1])
    matrix_y = _euler2matrix_y(euler[:, 1:2])
    matrix_z = _euler2matrix_z(euler[:, 2:3])

    matrix = matrix_x @ matrix_y @ matrix_z
    h = x @ matrix

    return h
