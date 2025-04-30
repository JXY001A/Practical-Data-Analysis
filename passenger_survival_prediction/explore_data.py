import pandas as pd

train_data = pd.read_csv('/Users/eleme/interview/Practical-Data-Analysis/decision_tree/passenger_survival_prediction/titanic_data/train.csv')
test_data = pd.read_csv('/Users/eleme/interview/Practical-Data-Analysis/decision_tree/passenger_survival_prediction/titanic_data/test.csv')

# 数据探索
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