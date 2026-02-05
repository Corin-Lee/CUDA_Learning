import numpy as np
import sys

def generate_softmax_test_data(n, c, dir='./'):
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
    data.tofile(dir + "input.bin")    
    expected_output.tofile(dir + "expected.bin")
    
    print(f"[test_data_gen.py]✅: Created input.bin and expected.bin (N={n}, C={c})")

if __name__ == '__main__':
    n = int(sys.argv[1])
    c = int(sys.argv[2])
    dir = sys.argv[3] 
    generate_softmax_test_data(n, c, dir)