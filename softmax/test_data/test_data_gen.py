import numpy as np

def generate_softmax_test_data(n, c):
    # 1. 随机生成输入数据 (包含大数值以测试稳定性)
    # 模拟一部分正常分布，一部分极端大值
    data = np.random.randn(n, c).astype(np.float32)
    data[0, 0] = 500.0  # 注入大数值
    data[0, 1] = -500.0 # 注入极小值

    # 2. 计算 NumPy 的真值 (稳定版本)
    # Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    shifted_data = data - np.max(data, axis=1, keepdims=True)
    exp_data = np.exp(shifted_data)
    expected_output = exp_data / np.sum(exp_data, axis=1, keepdims=True)

    # 3. 写入二进制文件供 C++ 读取
    data.tofile("input.bin")    
    expected_output.tofile("expected.bin")
    
    print(f"✅ Created input.bin and expected.bin (N={n}, C={c})")


generate_softmax_test_data(n=10, c=512)