import seaborn as sns
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

# sns.set(style="white")
# data = {"method":['Revised','Classic', 'Revised', 'Classic', 'Revised', 'Classic'],"Time":[4.86,8.798,5.55,11.97, 8.44, 107],"case":['case1', 'case1', 'case2', 'case2', 'case3', 'case3']}
# f1 = DataFrame(data)
# ax = sns.barplot(x="case", y="Time", hue="method", data=f1)
# bar_fig = ax.get_figure()
# bar_fig.savefig('./TimeComplexity', dpi = 400)