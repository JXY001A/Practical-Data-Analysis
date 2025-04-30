import pandas as pd
from sklearn.feature_extraction import DictVectorizer


train_data = pd.read_csv('/Users/eleme/interview/Practical-Data-Analysis/decision_tree/passenger_survival_prediction/titanic_data/train.csv')
test_data = pd.read_csv('/Users/eleme/interview/Practical-Data-Analysis/decision_tree/passenger_survival_prediction/titanic_data/test.csv')

# TODO: 数据探索
# 数据整体情况查看 
print(train_data.info())

# 数据总体描述
print('-'*30)
print(train_data.describe())

print('-'*30)

# 'O' 是Pandas中表示 object 类型的简写。
# 该参数指定仅统计数据类型为 object 的列（如字符串、分类变量）。
# 若省略 include 参数，默认仅统计数值型（int/float）列。

# 统计量	描述
# count	非空值的数量
# unique	唯一值的数量（去重后的类别数）
# top	出现频率最高的值（众数）
# freq	众数出现的频次
print(train_data.describe(include=['O']))

# 数据开头部分
print('-'*30)
print(train_data.head())

# # 数据结尾部分
print('-'*30)
print(train_data.tail())

# 计算 age 列的均值
print('Age mean',train_data['Age'].mean())


# TODO: 数据清洗
# 对所有 age 列中所有 NaN 列填充平均值
# inplace=True：直接修改原对象，不返回新对象。
# inplace=False（默认）：返回填充后的新对象，原对象不变。
# train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

# train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
# test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)


#各个港口的统计数量
print('港口聚合总数',train_data['Embarked'].value_counts())
# S    644
# C    168
# Q     77

#使用最多的港口 'S' 来填充缺省值
test_data['Embarked'].fillna('S')
test_data['Embarked'].fillna('S',inplace=True)

# TODO: 特征选择
# 通过数据探索我们发现，
# PassengerId为乘客编号，对分类没有作用，可以放弃；
# Name为乘客姓名，对分类没有作用，可以放弃；
# Cabin字段缺失值太多，可以放弃；
# Ticket字段为船票号码，杂乱无章且无规律，可以放弃。
# 其余的字段包括：Pclass、Sex、Age、SibSp、Parch和Fare，
# 这些属性分别表示了乘客的船票等级、性别、年龄、亲戚数量以及船票价格，
# 可能会和乘客的生存预测分类有关系。具体是什么关系，我们可以交给分类器来处理。

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# 训练特征提取
train_features = train_data[features]
# 是否存活标注
train_labels = train_data['Survived']
# 测试特征提取
test_features = test_data[features]

# 
dvec=DictVectorizer(sparse=False)

train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
# print(dvec.feature_names_)
# print(dvec.get_feature_names_out())
