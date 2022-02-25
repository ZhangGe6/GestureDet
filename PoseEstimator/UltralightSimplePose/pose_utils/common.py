import os
import torch

def delete_prevoius_checkpoint(arch, save_dir):
    # delete the previous checkpoints (with lower accuracy)
    if not os.path.exists(save_dir):
        return
        
    for file in os.listdir(save_dir):
        if file.startswith(arch):
            os.remove(os.path.join(save_dir, file))


def save_checkpoint(arch, state, epoch, acc1, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = '{}_epoch_{}_acc1_{:.2f}.pt'.format(
        arch, epoch, acc1
    )
    torch.save(state, os.path.join(save_dir, filename))