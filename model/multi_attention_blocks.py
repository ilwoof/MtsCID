import torch
import torch.nn as nn
from einops import rearrange
from model.attn_layer import AttentionLayer

class Inception_Attention_Block(nn.Module):
    def __init__(self, w_size, in_dim, d_model, patch_list=[10, 20], init_weight=True):
        super(Inception_Attention_Block, self).__init__()
        self.w_size = w_size
        self.in_dim = in_dim
        self.d_model = d_model
        self.patch_list = patch_list
        patch_attention_layers = []
        linear_layers = []
        for patch_size in self.patch_list:
            patch_number = w_size // patch_size
            patch_attention_layers.append(AttentionLayer(w_size=patch_number,
                                          d_model=patch_size,
                                          n_heads=1
                                                         )
                                          )
            linear_layers.append(nn.Linear(patch_number, patch_size))
        self.patch_attention_layers = nn.ModuleList(patch_attention_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, _, _ = x.size()
        res_list = []
        for i, p_size in enumerate(self.patch_list):
            z = rearrange(x, 'b (w p) c  -> (b c) w p', p=p_size).contiguous()
            _, z = self.patch_attention_layers[i](z)
            z = self.linear_layers[i](z)
            z = rearrange(z, '(b c) w p -> b (w p) c', b=B).contiguous()
            res_list.append(z)
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# Example usage
if __name__ == "__main__":
    kernel_sizes = [3, 60, 16]  # Example kernel sizes
    model = Inception_Attention_Block(w_size=60, in_dim=16, d_model=32)
    print(model)

    # Test the model with a random input
    input_tensor = torch.randn(3, 60, 32)  # Batch size of 5, 3 channels, 32 feature
    output = model(input_tensor)
    print(output.shape)  # Should be [1, 10] for 10 classes
