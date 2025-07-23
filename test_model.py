from models.dac_moe import MOEResidualVectorQuantize, DAC_MOE
import numpy as np
from functools import partial
# from dac.model.dac_encodec import DAC_ENCODEC
import torch
from audiotools import AudioSignal

if __name__ == '__main__':
    model = DAC_MOE()

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    # print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 44100 * 1
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")
