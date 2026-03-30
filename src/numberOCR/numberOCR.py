import processFiles
import utils

lr = 0.01

epochs = 100

dataset = []

# 加载数据集图片 准备灰度数组
processFiles.load_data(dataset)

# 加载测试图片 准备灰度数组
test_x = processFiles.process_bmp("src/numberOCR/test/2_0.bmp")

# 初始化网络参数
W1, b1, W2, b2 = utils.init_model()

# 训练模型
utils.train(dataset, W1, b1, W2, b2, epochs, lr)

# 验证训练集
print("训练集预测:")
for x, y in dataset:
    pred, probs = utils.predict(x, W1, b1, W2, b2)
    print(f"真实: {y}, 预测: {pred}, 概率: {[f'{p:.4f}' for p in probs]}")

# 测试
print("\n测试图片预测:")

pred, probs = utils.predict(test_x, W1, b1, W2, b2)

print(f"预测: {pred}, 概率: {[f'{p:.4f}' for p in probs]}")
        