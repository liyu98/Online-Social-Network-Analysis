# This is a sample Python script.

from pgmpy.factors.discrete import TabularCPD
import itertools
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

from matplotlib import pyplot as plt
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

import daft
import pandas as pd
from daft import PGM

from sklearn import tree
from sklearn.metrics import accuracy_score

import networkx as nx


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def assignment_a4():
    jpd_table = JPD(['A', 'B', 'C', 'D'], [2, 2, 2, 2],
                    [0.0448, 0.0252, 0.0112, 0.0588, 0.0144, 0.0144, 0.0096, 0.0216, 0.1024, 0.0576, 0.0256, 0.1344,
                     0.1152,
                     0.1152, 0.0768, 0.1728])

    p_a = jpd_table.marginal_distribution(['A'], inplace=False)
    print('P(A):')
    print(p_a)

    p_b = jpd_table.marginal_distribution(['B'], inplace=False)
    print('P(B):')
    print(p_b)

    p_ab = jpd_table.marginal_distribution(['A', 'B'], inplace=False)
    print('P(A,B):')
    print(p_ab)

    # A and B are not independent
    if p_ab != p_a * p_b:
        print("false")
    else:
        print("true")

    p_c = jpd_table.marginal_distribution(['C'], inplace=False)
    print('P(C):')
    print(p_c)

    p_ac = jpd_table.marginal_distribution(['A', 'C'], inplace=False)
    print('P(A,C):')
    print(p_ac)
    # A and C are not independent
    if p_ac != p_a * p_c:
        print("false")
    else:
        print("true")

    # given B:
    # If P(A,C|B) = P(A|B)*P(C|B)
    # P(A,C|B) * P(B) * P(B)  = P(A|B) * P(C|B) * P(B) * P(B)
    # This can be expanded to P(A,B,C) * P(B) == P(A,B) * P(B,C)

    phi_b = jpd_table.marginal_distribution(['B'], inplace=False).to_factor()

    phi_abc = jpd_table.marginal_distribution(['B', 'A', 'C'], inplace=False).to_factor()
    phi_ab = jpd_table.marginal_distribution(['B', 'A'], inplace=False).to_factor()
    phi_cb = jpd_table.marginal_distribution(['B', 'C'], inplace=False).to_factor()
    if phi_abc * phi_b != phi_ab * phi_cb:
        print('False')
    else:
        print('True')

    p_d = jpd_table.marginal_distribution(['D'], inplace=False)
    print('P(D):')
    print(p_d)

    # p_a_b = prob.marginal_distribution(['C'], None, ('B',),  condition_random_variable=True, inplace=False)
    print('P(A|B):')
    print(p_c)

    p_ad = jpd_table.marginal_distribution(['A', 'D'], inplace=False)
    print('P(A,D):')
    print(p_ad)
    # A and C are not independent
    if p_ad != p_a * p_d:
        print("false")
    else:
        print("true")

    # given C: P(A, D | C) = P(A | C) * P(D | C)

    # If P(A,D|C) = P(A|C)*P(D|C)
    # P(A,D|C) * P(C) * P(C)  = P(A|C)*P(D|C) * P(C) * P(C)
    # This can be expanded to P(A,D,C)*P(C) == P(A,C)*P(D,C)

    phi_c = jpd_table.marginal_distribution(['C'], inplace=False).to_factor()

    phi_adc = jpd_table.marginal_distribution(['C', 'A', 'D'], inplace=False).to_factor()
    phi_ac = jpd_table.marginal_distribution(['C', 'A'], inplace=False).to_factor()
    phi_dc = jpd_table.marginal_distribution(['C', 'D'], inplace=False).to_factor()
    if phi_adc * phi_c != phi_ac * phi_dc:
        print('False')
    else:
        print('True')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # construct Bayesian Network

    # Visualizing network structure
    pgm_play = PGM(shape=[5, 3])
    pgm_play.add_node(daft.Node('outlook', r"outlook", 1, 3))
    pgm_play.add_node(daft.Node('temp', r"temp", 2, 3))
    pgm_play.add_node(daft.Node('humidity', r"humidity", 4, 3))
    pgm_play.add_node(daft.Node('windy', r"windy", 5, 3))
    pgm_play.add_node(daft.Node('play', r"play", 3, 1))
    pgm_play.add_edge('temp', 'play')
    pgm_play.add_edge('outlook', 'play')
    pgm_play.add_edge('humidity', 'play')
    pgm_play.add_edge('windy', 'play')
    pgm_play.render()
    plt.show()

    # Defining Bayesian Structure
    data = {'outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy',
                        'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
            'temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                     'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
            'humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                         'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
            'windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True',
                      'False', 'False', 'False', 'True','True', 'False', 'True'],
            'play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes',
                     'Yes', 'Yes', 'Yes', 'Yes', 'No']}
    df = pd.DataFrame(data)
    print(df)

    # parameter estimation： 似然估计、贝叶斯估计
    # https: // github.com / pgmpy / pgmpy
    # 通过极大似然或者专家经验获取先验概率,选择计算MLE
    model = BayesianNetwork([('outlook', 'play'),
                             ('temp', 'play'),
                             ('humidity', 'play'),
                             ('windy', 'play')])

    mle = MaximumLikelihoodEstimator(model, df)
    print("mle:\n", mle.estimate_cpd('play'))

    print("mle-1:\n", mle.get_parameters())
    # mle = MaximumLikelihoodEstimator(model, data)
    # print(mle.estimate_cpd('fruit'))  # unconditional
    # print(mle.estimate_cpd('tasty'))  # conditional

    print("======================================测试使用==================")
    est = BayesianEstimator(model, df)
    print(est.estimate_cpd('play', prior_type='BDeu', equivalent_sample_size=10))
    # Setting equivalent_sample_size to 10 means
    # that for each parent configuration, we add the equivalent of 10 uniform samples

    # Calibrate all CPDs of model using MLE:
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    for cpd in model.get_cpds():
        print(cpd)

    # print(model.get_cpds('fruit'))
    # print(model.get_cpds('size'))
    # print(model.get_cpds('tasty'))

    #变量估计 make the inference https://www.cnblogs.com/bonelee/p/14312505.html

    infer = VariableElimination(model)

    print(infer.query(['play'], evidence={'outlook': 'Rainy', 'windy': 'True', 'humidity': 'Normal'}))

    print(infer.query(['play'], evidence={'temp': 'Hot', 'windy': 'False', 'humidity': 'Normal'}))

    # decision tree 数据量大，是个稀疏矩阵, 特征类别较多时，数据经过独热编码可能会变得过于稀疏。

    X = pd.get_dummies(df.iloc[:, :4])
    print(X)
    Y = df['play']
    print(Y)

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf = clf.fit(X, Y)
    tree.plot_tree(clf, filled=True, feature_names=X.columns)
    plt.show()

    # 特征重要性
    print(clf.feature_names_in_)

    print(clf.feature_importances_ )

    # https://github.com/pgmpy/pgmpy
    # accuracy_score(Y, clf.predict(X))

