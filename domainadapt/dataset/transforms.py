from medicaltorch.transforms import MTTransform
import numpy as np

class MTGaussianNoise(MTTransform):

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        noise = np.random.normal(self.mean, self.std, input_data.size)
        noise = noise.astype(np.float32)
        noise_img = input_data + noise
        rdict['input'] = input_data

        sample.update(rdict)
        return sample
