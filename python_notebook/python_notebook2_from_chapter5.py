#************************************************#************************************************
#************************************************#************************************************
#************************************************#************************************************
#第五章 关于pandas 库的 应用

from pandas import Series,DataFrame
import pandas as pd

obj=Series([4,7,-5,3])  #得到 索引+数据 的组合

obj2=Series([4,7,9,3],index=['a','b','c','d'])  #自己指定 索引

obj2[['a','c']]

#字典 Series 化
sdata={'r':512,'g':354,'o':348}
obj3=Series(sdata)


states=['g','o','d','t']

obj4=Series(sdata,index=states)

pd.isnull(obj4)

obj3+obj4  #相同索引 自动合并

#***********************************************************
#对DataFrame 的应用

data={'state':['a','a','b','c','d'],
      'year':['92','93','94','96','99'],
      'pop':[1.6,1.9,1.6,1.4,6.1]}

frame=DataFrame(data)

#按指定顺序 排列 列

DataFrame(data,columns=['pop','year','state','debt'],index=['1','2','3','4','5'])

frame.ix[3]

#补列

val=Series([-1.5,5.6,-4.3],index=['5','4','3'])

frame['debt']=val


#********************************************************
#字典的字典

pop={'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}

frame3=DataFrame(pop)

frame3.T




