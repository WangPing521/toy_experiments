import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('scheduler_curve')
begin_epoch = 10
max_epoch = 50

min_value = 0.2
max_value = 1.0

mult = -5

for i in range(100):
    if i < begin_epoch:
        weight = min_value
    elif i > max_epoch:
        weight = max_value
    else:
        weight = min_value + (max_value - min_value) * np.exp(
            mult * (1.0 - float(i - begin_epoch)/(max_epoch - begin_epoch)) ** 2
        )
    writer.add_scalar('weight_curve/lambda', weight, i)

