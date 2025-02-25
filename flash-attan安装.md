```bash
git clone --recurse-submodules https://github.com/HazyResearch/flash-attention.git
cd flash-attention
pip install ninja setuptools cmake
python setup.py install
```

验证是否安装成功：

```python
import torch
from flash_attn.flash_attention import FlashAttention

# 生成随机的矩阵数据（假设是自注意力计算的输入）
input_tensor = torch.randn(16, 128, 128, device="cuda")

# 使用 FlashAttention 执行计算
output = FlashAttention.apply(input_tensor, input_tensor, input_tensor)

print(output)
```



