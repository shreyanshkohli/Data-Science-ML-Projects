import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

sse = []
kmrange = range(1,10)

for i in kmrange:
    km = KMeans(n_clusters=i)
    km.fit(data)
    sse.append(km.inertia_)

sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', data=data, markers='o',color='r', label='sepal')
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', data=data, marker='*',color='g', label='petal')

plt.plot(kmrange,sse)

model = KMeans(n_clusters=3)
p = model.fit_predict(data)
data['predicted'] = p
data['truth'] = iris.target
score = model.score(data[['sepal length (cm)' , 'sepal width (cm)' , 'petal length (cm)' , 'petal width (cm)']], p)
print(data)

plt.style.use('ggplot')
# plt.show()