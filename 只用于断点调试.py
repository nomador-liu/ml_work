import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, max_iter=10000,
                 early_stopping=True, patience=100, tol=1e-5,
                 lambda_reg=0.01, random_state=None):
        """初始化Softmax回归模型"""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.weights = None  # 权重矩阵
        self.bias = None  # 偏置项
        self.classes = None  # 类别
        self.train_loss_ = []  # 训练损失记录
        self.val_loss_ = []  # 验证损失记录

    def _one_hot_encode(self, y):
        """将标签转换为独热编码"""
        n_samples = y.shape[0]
        one_hot = np.zeros((n_samples, self.n_classes))
        for i in range(n_samples):
            one_hot[i, y[i]] = 1
        return one_hot

    def _softmax(self, z):
        """计算softmax函数，添加数值稳定性改进"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def  _cross_entropy_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        n_samples = y_true.shape[0]
        epsilon = 1e-10  # 防止log(0)
        cross_entropy = -np.sum(y_true * np.log(y_pred + epsilon)) / n_samples

        # 添加L2正则化
        l2_reg = 0.5 * self.lambda_reg * (np.sum(self.weights ** 2) + np.sum(self.bias ** 2)) / n_samples

        return cross_entropy + l2_reg

    def _compute_gradients(self, X, y_true):
        """计算权重和偏置的梯度"""
        n_samples = X.shape[0]

        # 前向传播
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(z)

        # 计算误差
        error = y_pred - y_true

        # 计算梯度
        dw = (np.dot(X.T, error) + self.lambda_reg * self.weights) / n_samples
        db = (np.sum(error, axis=0) + self.lambda_reg * self.bias) / n_samples

        return dw, db, self._cross_entropy_loss(y_pred, y_true)

    def fit(self, X, y, val_size=0.2):
        """训练模型"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 获取类别信息
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        n_samples, n_features = X.shape

        # 划分训练集和验证集
        shuffle_idx = np.random.permutation(n_samples)
        val_idx = shuffle_idx[:int(n_samples * val_size)]
        train_idx = shuffle_idx[int(n_samples * val_size):]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 转换标签为独热编码
        y_train_onehot = self._one_hot_encode(y_train)
        y_val_onehot = self._one_hot_encode(y_val)

        # 初始化参数
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)

        # 早停机制相关变量
        best_loss = np.inf
        best_weights = None
        best_bias = None
        no_improve_count = 0

        # 批量梯度下降
        for i in range(self.max_iter):
            # 计算梯度和训练损失
            dw, db, train_loss = self._compute_gradients(X_train, y_train_onehot)
            self.train_loss_.append(train_loss)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 计算验证损失
            z_val = np.dot(X_val, self.weights) + self.bias
            y_val_pred = self._softmax(z_val)
            val_loss = self._cross_entropy_loss(y_val_pred, y_val_onehot)
            self.val_loss_.append(val_loss)

            # 早停检查
            if self.early_stopping:
                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.patience:
                        print(f"早停于迭代 {i + 1}，最佳验证损失: {best_loss:.6f}")
                        self.weights = best_weights
                        self.bias = best_bias
                        break

        return self

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.weights) + self.bias
        return self._softmax(z)

    def predict(self, X):
        """预测类别"""
        y_proba = self.predict_proba(X)
        return self.classes[np.argmax(y_proba, axis=1)]

    def plot_decision_boundary(self, X, y, feature_indices=[0, 1], title="决策边界"):
        """
        绘制决策边界

        参数:
            X: 特征数据
            y: 标签数据
            feature_indices: 用于绘图的两个特征的索引
            title: 图表标题
        """
        if len(feature_indices) != 2:
            raise ValueError("请指定两个特征的索引用于绘制决策边界")

        # 只使用指定的两个特征进行可视化
        X_plot = X[:, feature_indices]
        feature_names = iris.feature_names
        feature1_name = feature_names[feature_indices[0]]
        feature2_name = feature_names[feature_indices[1]]

        # 设置网格范围
        h = 0.02  # 网格步长
        x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
        y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 创建完整特征向量（其他特征使用均值）
        n_features = X.shape[1]
        grid_points = np.zeros((xx.ravel().shape[0], n_features))
        # 填充均值到其他特征
        for i in range(n_features):
            if i in feature_indices:
                idx = feature_indices.index(i)
                grid_points[:, i] = [xx.ravel(), yy.ravel()][idx]
            else:
                grid_points[:, i] = np.mean(X[:, i])

        # 预测网格点的类别
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # 设置颜色
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # 绘制决策边界
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

        # 绘制数据点
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel(feature1_name, fontsize=12)
        plt.ylabel(feature2_name, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(iris.target_names, loc='best')
        plt.show()

    def plot_loss_curve(self):
        """绘制训练损失和验证损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.train_loss_)), self.train_loss_, label='训练损失')
        plt.plot(range(len(self.val_loss_)), self.val_loss_, label='验证损失')
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('损失值', fontsize=12)
        plt.title('训练损失和验证损失曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# 主函数测试
if __name__ == "__main__":
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data  # 特征
    y = iris.target  # 标签

    # 数据标准化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"数据集划分: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    print(f"类别: {iris.target_names}")
    print(f"特征: {iris.feature_names}")

    # 创建并训练模型
    model = SoftmaxRegression(
        learning_rate=0.1,
        max_iter=10000,
        early_stopping=True,
        patience=100,
        lambda_reg=0.01,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 绘制损失曲线
    model.plot_loss_curve()

    # 预测与评估
    y_pred = model.predict(X_test)

    print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 绘制不同特征组合的决策边界
    # 使用花瓣长度和花瓣宽度 (索引2和3)
    model.plot_decision_boundary(
        X, y,
        feature_indices=[2, 3],
        title="基于花瓣长度和花瓣宽度的决策边界"
    )

    # 使用花萼长度和花萼宽度 (索引0和1)
    model.plot_decision_boundary(
        X, y,
        feature_indices=[0, 1],
        title="基于花萼长度和花萼宽度的决策边界"
    )
