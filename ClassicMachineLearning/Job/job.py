# Decision tree

import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

###
df_treino = pd.DataFrame({
    'salary': ['high', 'small', 'small', 'high', 'high', 'small'],
    'localization': ['away', 'near', 'away', 'away', 'near', 'away'],
    'function': ['interest', 'disinterest', 'interest', 'disinterest', 'interest', 'disinterest'],
    'decision': ['sim', 'nao', 'sim', 'nao', 'sim', 'nao']
})

print(df_treino)

x_train = df_treino.iloc[:, :-1]
y_train = df_treino.iloc[:, -1]

# Variables to be substituted in the dataset
high = 1
small = 0
near = 1
away = 0
interest = 1
disinterest = 0

x_train = x_train.replace(
    {
        'high': high,
        'small': small,
        'near': near, 'away': away,
        'interest': interest, 'disinterest': disinterest
    }
)

# Build decision tree
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

plot_to_show = tree.plot_tree(model)

plt.show()
