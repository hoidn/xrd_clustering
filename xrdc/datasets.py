import numpy as np
from siamese_xrd import train

def ndarr_to_dataloader(X, Y):
    tensor_x = torch.Tensor(X[:, None, :]) # transform to torch tensor
    tensor_y = torch.Tensor(Y[:, None])
    my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    
    dataloader = DataLoader(my_dataset) # create your dataloader
    return dataloader

def dataloader_to_ndarr(original_test_loader):
    X0, y0 = zip(*list(original_test_loader))
    X0, y0 = np.vstack(X0).squeeze(), np.hstack(y0)
    return X0, y0

# TODO define a RunOutput class

#def load_run_data(name):
#    torch.save(net, add_prefix('model.serialized'))
#    torch.save(original_test_loader, add_prefix('original_test_loader.serialized'))
