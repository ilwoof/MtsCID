import torch
import torch.nn as nn


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_list=[1, 3, 5], groups=1, init_weight=True):
        super(Inception_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_list = kernel_list
        kernels = []
        for i in self.kernel_list:
            kernels.append(nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=i,
                                     padding='same',
                                     padding_mode='circular',
                                     bias=False,
                                     groups=groups))
        self.convs = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(len(self.kernel_list)):
            res_list.append(self.convs[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# Example usage
if __name__ == "__main__":
    kernel_sizes = [3, 5, 7]  # Example kernel sizes
    model = Inception_Block(in_channels=3,
                            out_channels=6,
                            kernel_list=kernel_sizes,
                            groups=3
                            )
    print(model)

    # Test the model with a random input
    input_tensor = torch.randn(5, 3, 32)  # Batch size of 5, 3 channels, 32 feature
    output = model(input_tensor)
    print(output.shape)  # Should be [1, 10] for 10 classes
