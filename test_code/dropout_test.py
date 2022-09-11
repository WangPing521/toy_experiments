import numpy as np


def dropout(x, level):
    if level < 0. or level >= 1:
        raise ValueError('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level

    random_tensor = np.random.binomial(n=1, p=retain_prob, size=x.shape) #即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    print(random_tensor)

    x *= random_tensor
    print(x)
    x /= retain_prob

    return x


x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
dropout(x,0.4)