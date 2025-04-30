from sklearn.feature_extraction import DictVectorizer


data = [
    {'Age': 25, 'City': 'Beijing', 'Gender': 'Male'},
    {'Age': 30, 'City': 'Shanghai', 'Gender': 'Female'},
    {'Age': 28, 'City': 'Beijing', 'Gender': 'Male'}
]


dvec = DictVectorizer(sparse=False)
X = dvec.fit_transform(data)

print("特征矩阵:\n", X)
print("特征名称:\n", dvec.get_feature_names_out())