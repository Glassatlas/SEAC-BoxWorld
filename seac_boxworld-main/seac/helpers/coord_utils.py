import numpy as np
import torch


def get_coordinate_grid(batch_size: int, height: int, width: int) -> torch.Tensor:
    xmap = np.linspace(-np.ones(height), np.ones(height), num=width, endpoint=True, axis=0)
    xmap = torch.tensor([[xmap]], dtype=torch.float32, requires_grad=False)
    ymap = np.linspace(-np.ones(width), np.ones(width), num=height, endpoint=True, axis=1)
    ymap = torch.tensor([[ymap]], dtype=torch.float32, requires_grad=False)

    coor = torch.cat((xmap, ymap), dim=1)
    coor = torch.tile(coor, [batch_size, 1, 1, 1])

    return coor


def get_coordinate_grid2(batch_size: int, height: int, width: int) -> torch.Tensor:
    """
    The output of cnn is tagged with two extra channels indicating the spatial position(x and y) of each cell

    :param input_tensor: (TensorFlow Tensor)  [B,Height,W,D]
    :return: (TensorFlow Tensor) [B,Height,W,2]
    """

    coor = [[[h / height, w / width] for w in range(width)] for h in range(height)]
    # coor = []
    # for h in range(height):
    #     w_channel = []
    #     for w in range(width):
    #         w_channel.append([float(h / height), float(w / width)])
    #     coor.append(w_channel)
    coor = torch.tensor([coor]).movedim(3, 1)
    # [1,Height,W,2] --> [B,Height,W,2]
    coor = torch.tile(coor, [batch_size, 1, 1, 1])
    return coor
