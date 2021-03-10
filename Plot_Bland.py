import seaborn as sns
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

# sns.set(style="white")
# data = {"Rule":['Normal','Bland', 'Normal','Bland', 'Normal','Bland'],"Time":[7.89,8.79,9.66,11.97, 88.26, 107],"case":['case1', 'case1', 'case2', 'case2', 'case3', 'case3']}
# f1 = DataFrame(data)
# ax = sns.barplot(x="case", y="Time", hue="Rule", data=f1)
# bar_fig = ax.get_figure()
# bar_fig.savefig('./Bland', dpi = 400)

sns.set(style="white")
data = {"Rule":['Normal','Bland', 'Normal','Bland', 'Normal','Bland'],"Number of Pivoting":[5,3,7,5, 24, 26],"case":['case1', 'case1', 'case2', 'case2', 'case3', 'case3']}
f1 = DataFrame(data)
ax = sns.barplot(x="case", y="Number of Pivoting", hue="Rule", data=f1)
bar_fig = ax.get_figure()
bar_fig.savefig('./Bland_Pivoting', dpi = 400)