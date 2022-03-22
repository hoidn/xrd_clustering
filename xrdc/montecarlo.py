import numpy as np

def sample(signal_estimate, noise_estimate, N = 100):
    """
    Given data matrix and matrix of estimates of the noise standard deviation,
    ............ todo
    """
    gaussian_noise = np.random.normal(size = N * np.prod((signal_estimate.shape))).reshape((N,)\
                                                                        + signal_estimate.shape)
    noise_samples = noise_estimate * gaussian_noise
    #return noise_samples
    return signal_estimate + noise_samples
