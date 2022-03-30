import torch
import torch.nn as nn
import numpy as np

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, true, generate): # true and generate default as a list
        n_classes = true[0].shape[1]
        dice = np.zeros(n_classes)
        dice_tp = np.zeros(n_classes)
        dice_all = np.zeros(n_classes)
        for i in range(len(true)):
            generate_classes = torch.argmax(generate[i], 1)  # tensor([2])
            true_classes = torch.argmax(true[i], 1)  # tensor([2])
            for c in range(n_classes):
                dice_tp[c] += ((generate_classes == c) & (true_classes == c)).sum().item()
                dice_all[c] += ((true_classes == c).sum().item())
                dice[c] = dice_tp[c] / dice_all[c]

        return dice[:]  # omit the background channel

if __name__ == '__main__':
    pass