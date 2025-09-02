import struct
import numpy as np
import gzip
from pathlib import Path
import sys
def add(x, y):
    """一个简单的add函数，以便熟悉自动测试（pytest）

    Args:
        x (Python数字 或者 numpy array)
        y (Python数字 或者 numpy array)

    Return:
        x+y的和
    """
    ### 你的代码开始
    return x+y
    ### 你的代码结束


def parse_mnist(image_filename, label_filename):
    """ 读取 MNIST 格式的图像和标签文件。有关文件格式的说明，请参阅此页面：
    http://yann.lecun.com/exdb/mnist/。

    参数：
    image_filename（字符串）：MNIST 格式的 gzip 压缩图像文件的名称
    label_filename（字符串）：MNIST 格式的 gzip 压缩标签文件的名称

    返回：
    tuple (X,y)：
    x (numpy.ndarray[np.float32])：包含已加载数据的二维 numpy 数组。数据的维度应为
    (num_examples x input_dim)，其中“input_dim”是数据的完整维度，例如，由于 MNIST 图像为 28x28，因此
    input_dim 为 784。值应为 np.float32 类型，并且数据应被归一化为最小值为 0.0，
    最大值为 1.0 （即将原始值 0 缩放为 0.0，将 255 缩放为 1.0）。

    y (numpy.ndarray[dtype=np.uint8])：包含示例标签的一维 NumPy 数组。值应为 np.uint8 类型，对于 MNIST，将包含 0-9 的值。
    """

    ### 你的代码开始
    # Python中的struct模块（以及gzip模块，当然还有numpy本身）来实现此函数。
    # gzip.open打开文件，先解析前16比特，会返回mnist魔数2051，样本数60000，样本形状28，28
    # image_filename = Path("../" +image_filename)
    # label_filename = Path("../" +label_filename)
    sys.path.append("..")
    with gzip.open(image_filename, 'rb') as f:
        image_header = struct.unpack(">IIII", f.read(16))
        magic_num, num_images, rows, cols = image_header
        X = struct.unpack(f">{num_images*rows*cols}B", f.read())
    # gzip.open打开文件，先解析前8比特，会返回mnist魔数2049，样本数60000，
    with gzip.open(label_filename, 'rb') as f:
        _ = struct.unpack(">II", f.read(8))
        y = struct.unpack(f">{num_images}B", f.read())
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.uint8)
    X = X.reshape(num_images, rows*cols)

    return X,y
    ### 你的代码结束


def softmax_loss(Z, y):
    """ 返回 softmax 损失。

    参数：
    z (np.ndarray[np.float32])：形状为 (batch_size, num_classes) 的二维 NumPy 数组，
    包含每个类别的 对数概率 预测值 （softmax函数激活之前的值）。

    y (np.ndarray[np.uint8])：形状为 (batch_size, ) 的一维 NumPy 数组，包含每个样本的真实标签。

    返回：
    样本的平均 softmax 损失。
    """

    ### 你的代码开始
    # 先将y展成稀疏矩阵
    y_values = np.unique(y).reshape(1,-1)
    y_sparse = y_values == y.reshape(-1,1)
    y_sparse = y_sparse.astype(np.int8)

    # 计算损失
    # 1.z各个数值进行exp,生成概率
    z_exp = np.exp(Z)
    Z_proba = z_exp / np.sum(z_exp,axis=1,keepdims=True)

    # 2. 计算损失 损失
    loss = -np.sum(y_sparse * np.log(Z_proba)) / len(y)
    return loss

    ### 你的代码结束


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ 使用步长 lr 和指定的批次大小，对数据运行单轮 小批量梯度下降 进行 softmax 回归。此函数会修改
    θ 矩阵，并迭代 X 中的批次，但不对顺序进行随机化。

    参数：
    X (np.ndarray[np.float32])：大小为
    (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    theta (np.ndarrray[np.float32])：softmax 回归的二维数组参数，形状为 (input_dim, num_classes)。
    lr (float)：SGD 的步长（学习率）。
    batch (int)：SGD 小批次的大小。

    返回：
    无
    """

    ### 你的代码开始
    num_examples = len(X)
    # 先将y展成稀疏矩阵
    # y_values = np.unique(y).reshape(1, -1)
    # y_sparse = y_values == y.reshape(-1, 1)
    # y_sparse = y_sparse.astype(np.int8)
    for start in range(0, num_examples, batch):
        X_batch = X[start:start+batch,:]
        m = X_batch.shape[0]
        # y_batch = y_sparse[start:start+batch,:]
        y_batch = np.zeros(shape = (m,theta.shape[1]))
        y_batch[np.arange(m), y[start:start+batch]] = 1

        Z_ = np.exp(X_batch@theta)
        Z = Z_/np.sum(Z_,axis=1,keepdims=True)

    # 计算梯度与梯度更新
        dec_theta = X_batch.T@(Z-y_batch) / m  # n,m    m,k  -> n,k
        theta -= lr*dec_theta


    ### 你的代码结束


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ 对由权重 W1 和 W2 定义的双层神经网络（无偏差项）运行一个 小批量梯度下降 迭代轮次：
    logits = ReLU(X * W1) * W2
    该函数应使用步长 lr 和指定的批次大小（并且同样，不随机化 X 的顺序）。它应修改 W1 和 W2 矩阵。

    参数：
    X (np.ndarray[np.float32])：大小为 (num_examples x input_dim) 的二维输入数组。
    y (np.ndarray[np.uint8])：大小为 (num_examples,) 的一维类别标签数组。
    W1 (np.ndarray[np.float32])：第一层权重的二维数组，形状为(input_dim, hidden_dim)
    W2 (np.ndarray[np.float32])：第二层权重的二维数组。形状
    (hidden_dim, num_classes)
    lr (float)：SGD 的步长（学习率）
    batch (int)：SGD 小批次的大小

    返回：
    无
    """

    ### 你的代码开始
    num_examples = len(X)
    for start in range(0, num_examples, batch):
        X_batch = X[start:start+batch,:]
        m = X_batch.shape[0]
        y_batch = np.zeros(shape=(m, W2.shape[1]))
        y_batch[np.arange(m), y[start:start + batch]] = 1

        A1 = X_batch@W1                 # (m, input_dim) @ ( input_dim, hidden_dim) = (m , hidden_dim)
        Z1 = np.maximum(A1, 0)          # (m , hidden_dim)
        # A2 = np.exp(Z1@W2)
        A2 = Z1 @ W2  # (m, hidden_dim) @ (hidden_dim, num_classes) = (m, num_classes)
        A2 -= np.max(A2, axis=1, keepdims=True)
        A2 = np.exp(A2)
        Z2 = A2 / np.sum(A2, axis=1, keepdims=True)  # Softmax输出
        G2 = Z2-y_batch                                 # (m,num_classes) - (m,num_classes) = (m,num_classes)
        G1 = np.where(Z1>0,1.0,0.0)*(G2@W2.T)               # (m , hidden_dim)* [(m,num_classes)@(hidden_dim, num_classes).T ] = (m , hidden_dim)* (m,hidden_dim) =(m,hidden_dim)
        des_W1 = X_batch.T@G1                          # (m, input_dim).T @ (m,hidden_dim) = (input_dim,hidden_dim)
        des_W2 = Z1.T@G2                                # (m , hidden_dim).T @ (m,num_classes) = (hidden_dim,num_classes)
        W1 -= lr*des_W1/m
        W2 -= lr*des_W2/m

    ### 你的代码结束

### 下面的代码不用编辑，只是用来展示功能的

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100):
    """ 示例函数，用softmax回归训练 """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ 示例函数，训练神经网络 """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":

    X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)