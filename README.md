# QDU_Python2024
青岛大学2024年zdj老师python实验课作业

## 24_1.1
'''
    定义函数 drawdouble(n, i)
        n1 = n / 2
        a = (180 * (n - 2) / n)
        b = 180 - a
        c = b / 2
        d = 180 - (c * (n / 2 - 1))
        d1 = 将 d 转换为弧度
        c1 = 将 c 转换为弧度
        e = (sin(c1) / sin(d1)) * 500
        当 i < n1 时重复
            前进 500
            左转 90 度
            抬起画笔
            前进 e
            放下画笔
            左转 90 度
            前进 500
            左转 180 - 180 / n1
            i 增加 1
        结束当
    结束函数

    定义函数 drawsingle(n)
        放下画笔
        开始填充
        重复 n 次
            前进 500
            左转 180 - 180 / n
        结束重复
        结束填充
    结束函数

    n = 输入整数("请输入星星的角数: ")
    i = 0  # 初始化循环变量
    如果 n 是奇数
        调用 drawsingle(n)
    否则
        调用 drawdouble(n, i)'''

## 24_1.2
'''
    从 turtle 导入所有功能

        重复 100 次，计数变量 i 从 0 到 99
            设置海龟的朝向为 90 * i + 90 度
            前进 10 + 5 * i 的距离
        结束重复

        完成绘图'''

## 24_2

'''
    定义函数 koch(size, n)
        如果 n 等于 0
            向前移动 size
        否则
            对于 angle 在 [0, 60, -120, 60] 中的每一个
                左转 angle 度
                递归调用 koch(size/3, n-1)
            结束对于
        结束如果
    结束函数

    定义函数 main()
        设置海龟速度为 10000
        设置窗口大小为 600x600
        抬起画笔
        移动到坐标 (-200, 100)
        放下画笔
        设置画笔宽度为 2
        设置绘图层级 level 为 3
        重复 5 次
            调用 koch(400, level)
            随机生成 90 到 270 度的转角 turn_angle
            右转 turn_angle 度
        结束重复
        隐藏海龟
        结束绘图
    结束函数

    调用 main()
'''

## 24_3

'''
    导入 jieba 库
    从 wordcloud 导入 WordCloud

    打开文件 '24_3/关于实施乡村振兴的意见.txt'，使用 'utf-8' 编码读取
        读取文件内容到变量 text 中
    关闭文件

    # 使用 jieba 进行中文分词
    将 text 进行分词，结果存入变量 words
    将 words 中的词用空格连接成字符串

    创建词云对象 wordcloud，参数如下：
        字体路径为 '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        宽度为 800 像素
        高度为 400 像素
        背景颜色为白色
        生成词云图，使用 words 作为输入

    # 保存词云图
    将词云图保存到文件 '24_3/wordcloud.png'

    # 显示词云图
    导入 matplotlib.pyplot 库并命名为 plt
    使用 plt 显示词云图，插值方式为 'bilinear'
    关闭坐标轴显示
    显示图像
'''
## 24_4
### 24_4
'''
    导入必要的库
    从 os 导入 pipe
    从 matplotlib.offsetbox 导入 DrawingArea
    导入 torch 及其子模块
    导入 torchvision 的 datasets 和 transforms 模块
    从 torch 导入 optim
    从 torch.utils.data 导入 DataLoader

    # 定义超参数
    BATCH_SIZE = 256
    DEVICE = 如果 GPU 可用则使用 "cuda"，否则使用 "cpu"
    EPOCHS = 100

    # 构建数据处理管道
    pipeline = 将以下转换组合起来：
        将图片转换为 tensor
        正则化处理，均值为 0.1307，标准差为 0.3081

    # 下载并加载数据集
    train_set = 下载并加载 MNIST 训练集，应用转换 pipeline
    test_set = 下载并加载 MNIST 测试集，应用转换 pipeline

    # 加载数据到 DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    # 显示 MNIST 中的图片
    打开文件 './data/MNIST/raw/t10k-images-idx3-ubyte' 并读取内容
    将文件内容转换为整数列表 image1，从偏移 16 开始，长度为 784
    打印 image1

    将 image1 转换为 numpy 数组 image1_np，数据类型为 uint8，形状为 28x28x1
    打印 image1_np 的形状

    使用 OpenCV 保存 image1_np 为图像文件 '24_4/digit.jpg'

    # 构建神经网络模型
    定义类 Digit 继承自 nn.Module
        定义初始化方法
            初始化父类
            定义卷积层 conv1，输入通道 1，输出通道 10，卷积核大小 5x5
            定义卷积层 conv2，输入通道 10，输出通道 20，卷积核大小 3x3
            定义全连接层 fcl，输入大小 20x10x10，输出大小 500
            定义全连接层 fcl2，输入大小 500，输出大小 10
        定义前向传播方法
            获取输入的批量大小
            使用 conv1 进行卷积
            使用 ReLU 激活
            使用 2x2 池化层进行池化
            使用 conv2 进行卷积
            展平输入
            使用 fcl 进行全连接层运算
            使用 ReLU 激活
            使用 fcl2 进行全连接层运算
            使用 log_softmax 计算分类概率
            返回输出

    # 定义优化器
    初始化模型 Digit，并将其移动到 DEVICE 上
    定义优化器为 Adam，传入模型参数

    # 定义训练方法
    定义函数 train_model(model, device, train_loader, optimizer, epoch)
        设定模型为训练模式
        遍历 train_loader 中的批次数据
            将数据和标签移动到 DEVICE
            初始化优化器梯度为 0
            前向传播计算输出
            计算损失
            反向传播
            优化器更新参数
            每 3000 个批次打印损失

    # 定义测试方法
    定义函数 test_model(model, device, test_loader)
        设定模型为评估模式
        初始化正确预测计数 correct 和测试损失 test_loss
        禁用梯度计算
        遍历 test_loader 中的批次数据
            将数据和标签移动到 DEVICE
            前向传播计算输出
            累积测试损失
            计算预测值
            计算正确预测数量
        打印平均损失和准确率

    # 训练和测试模型
    对于每一个 epoch 从 1 到 EPOCHS
        调用 train_model 训练模型
        调用 test_model 测试模型

    # 保存模型到文件
    定义模型保存路径 model_save_path 为 'model.pth'
    保存模型状态字典到 model_save_path
    打印模型已保存的信息
'''

### 24_4_apl
这是针对静态图像的应用示例
‘’‘
    导入必要的库
    从 cv2 导入
    从 numpy 导入
    从 torch 导入
    从 torchvision 导入 transforms
    导入 torch.nn 作为 nn
    导入 torch.nn.functional 作为 F
    从 DigitModel 导入 Digit 类

    设置设备 DEVICE 为 GPU 如果可用，否则为 CPU

    # 加载训练好的模型
    初始化 Digit 模型，并将其移动到 DEVICE
    加载模型参数 state_dict 从 'model.pth'
    将模型设置为评估模式

    # 图片预处理函数
    定义函数 preprocess_image(image_path)
        读取灰度图片
        将图片调整为 28x28 大小
        将图片像素值归一化到 [0, 1] 之间
        将图片转换为 tensor
        增加维度使其适应模型输入形状
        对图片 tensor 进行正则化处理
        返回预处理后的图片 tensor
    结束函数

    # 预测函数
    定义函数 predict(model, image_tensor)
        将图片 tensor 移动到 DEVICE
        使用模型进行前向传播，得到输出
        计算输出中最大值的索引
        返回预测结果
    结束函数

    # 进行图片预处理和预测
    定义图片路径 image_path 为 'OIP-C.jpeg'
    调用 preprocess_image 函数对图片进行预处理，得到 image_tensor
    调用 predict 函数进行预测，得到 predicted_digit
    打印预测结果
’‘’

### 24_4_video
这是针对视频的使用示例
'''
    导入必要的库
    从 cv2 导入
    从 torch 导入
    从 torchvision 导入 transforms
    导入 torch.nn 作为 nn
    导入 torch.nn.functional 作为 F
    从 DigitModel 导入 Digit 类

    设置设备 DEVICE 为 GPU 如果可用，否则为 CPU

    # 加载训练好的模型
    初始化 Digit 模型，并将其移动到 DEVICE
    加载模型参数 state_dict 从 'model.pth'
    将模型设置为评估模式

    # 图片预处理函数
    定义函数 preprocess_image(img)
        将图片调整为 28x28 大小
        将图片像素值归一化到 [0, 1] 之间
        将图片转换为 tensor
        增加维度使其适应模型输入形状
        对图片 tensor 进行正则化处理
        返回预处理后的图片 tensor
    结束函数

    # 预测函数
    定义函数 predict(model, image_tensor)
        将图片 tensor 移动到 DEVICE
        使用模型进行前向传播，得到输出
        计算输出中最大值的索引
        返回预测结果
    结束函数

    # 打开摄像头
    调用 cv2.VideoCapture(0) 打开默认摄像头，保存为 cap

    # 无限循环处理摄像头帧
    当条件为真时重复
        读取摄像头帧，保存为 ret 和 frame
        将 frame 转换为灰度图像 gray_frame
        对灰度图像 gray_frame 进行预处理，得到 preprocessed_frame
        使用模型进行预测，得到 prediction
        在 frame 上显示预测结果
        显示处理后的 frame
        如果按下空格键，则退出循环
    结束循环

    释放摄像头 cap
    关闭所有 OpenCV 窗口
'''