# PyTorch Notes
- `torch.compile`: decorator
- `TorchDynamo`: a Python bytecode analyzer that traces a function and captures the computation graph
- `TorchInductor`: compiler backend that takes the graph and generates a Triton kernel that fuses operations together
- The tradeoff is between control and convenience.
- `torch.profiler`

```python3
from torch.profiler import profile, ProfilerActivity
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # launch kernels
    torch.cuda.synchronize()
prof.export_chrome_trace("trace.json")
```
