import matplotlib.pyplot as plt
import seaborn as sb

df = sb.load_dataset("tips")
print(df.head(5))  # here tips column is not in final.

X = df.drop('tip',axis=1)
y = df['tip']
print(X,y)
print(df.corr()) #corr only returns numeric value no categories

#sb.heatmap(df.corr()) # similar to confusionmatrix display
#plt.show()

#sb.jointplot(x='tip',y='total_bill',data=df,kind='reg') # get histogram o/p #kind ='hex' for hexagonal points
#plt.show()

#sb.pairplot(df,hue='sex')
#plt.show()

#sb.displot(df['tip'])
#plt.show()

x = [1,1,2,2,2,2,25,56,9]
#plt.hist(x)
#plt.show()
#sb.countplot('sex',data=df,hue='smoker')
#sb.countplot(y='sex',data=df) i get plot from y axis , hue='time'
#plt.show()
#sb.barplot(x='total_bill',y='day',data=df)
#sb.boxplot('day','total_bill',data=df)
sb.violinplot(x='total_bill',y='day',data=df)
plt.show()