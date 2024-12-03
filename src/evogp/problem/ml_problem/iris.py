import jax.numpy as jnp

from ..problem import Problem

from evogp.cuda.operations import forward, sr_fitness

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target
X, y
# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建KNN模型
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 评估模型
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# print(report)
# acc

class IRIS(Problem):
    def __init__(self):
        super().__init__()
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
        self.data_inputs = jnp.array(X)
        self.data_outputs = jnp.array(y)

    def evaluate(self, randkey, trees):
        res = sr_fitness(
            trees,
            data_points=self.inputs.astype(jnp.float32),
            targets=self.targets.astype(jnp.float32),
        )

        # import jax
        # pop_size = trees.shape[0]
        # res = jax.random.uniform(randkey, (pop_size,))

        return -res

    def show(self, randkey, prefix_trees):
        pass
        # predict = act_func(state, self.inputs, params)
        # inputs, target, predict = jax.device_get([self.inputs, self.targets, predict])
        # loss = -self.evaluate(randkey, state, act_func, params)
        # msg = ""
        # for i in range(inputs.shape[0]):
        #     msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        # msg += f"loss: {loss}\n"
        # print(msg)

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return self.data_inputs.shape

    @property
    def output_shape(self):
        return self.data_outputs.shape