# coding=utf8
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures  # 导入多项式回归模型
import numpy as np


# 线性回归-->
def makeLinear(x, y, fit_intercept=False, normalize=False, n_jobs=None, ifSave=False):
    '''
    :param x: 训练集x
    :param y: 训练集y
    :param fit_intercept: 是否计算截距。False - 模型没有截距
    :param normalize: 当fit_intercept设置为False时，该参数将被忽略。
                        如果为真，则回归前的回归系数X将通过减去平均值并除以l2 - 范数而归一化。
    :param n_jobs: 指定线程数
    :param ifSave: 是否保存模型，如是，指定保存的文件名
    :return:
    '''
    lrModel = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs)
    lrModel.fit(x, y)
    #
    # 打印截距和系数
    intercept = lrModel.intercept_
    coef = lrModel.coef_
    print('截距:{}；参数:{}；模型得分:{}'.format(intercept, coef, lrModel.score(x, y)))

    # 保存模型，方便下次调用
    if ifSave:
        import pickle
        with open('model/{}.pkl'.format(ifSave), 'wb') as f:
            pickle.dump(lrModel, f, -1)
        print('模型保存结束')


def useLinear(x, modelName):
    import pickle
    with open('model/{}.pkl'.format(modelName), 'rb') as f:
        model = pickle.load(f)
        y = model.predict(x)
        print('预测结果', y)


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    x = [[0, 0], [1, 1.1], [2, 2]]
    y = [[0], [1], [2]]

    # calurateLinear(x, y,ifSave='demo')
    useLinear(x, 'demo')
