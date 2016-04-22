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
#***********************************************************
from pandas import Series,DataFrame
import pandas as pd


#DataFrame 中的index 可以看作是“行号”
           # 而columns 可以看作是“列类”


#DataFrame 中和的index 是不可修改的 immutable


#P126 介绍了 index 的许多方法

#append  连接另一个Index对象，产生一个新的Index
#intersection 计算交集
#union 计算并集
#delete 删除索引i处的的元素，并得到新的index
#insert 将元素插入到索引i处，并得到新的index



#****************************************************
p126


reindex 重新排布index 缺失补NaN

obj=Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])

obj.reindex(['a','b','c','d','e'],fill_value=0) # fill_value: 用该值来代替NaN

#向上 填充NaN

obj3=Series(['blue','purple','yellow'],index=[0,2,4])

obj3.reindex(range(6),method='ffill')

#*********************************************************
frame=DataFrame(np.arange(9).reshape((3,3)),index=['a','c','d'],columns=['Ohio','Texas','California'])

frame2=frame.reindex(['a','b','c','d'])



#丢弃 drop  删除 索引值

obj=Series(np.arange(5),index=['a','b','c','d','e'])

new_obj=obj.drop('c')

data=DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])

#删除 列类型

data.drop('three',axis=1)

#直接 定位到元素 点 用ix
data.ix['Colorado',['one','four']]


#***************************
#对Series 相加  是对相同索引 的数据相加，没有的值 其和 最终会以NaN 来表示
list('abcd') 等价于 ['a','b','c','d']

df1=DataFrame(np.arange(12).reshape(3,4),columns=list('abcd'))

df2=DataFrame(np.arange(20).reshape(4,5),columns=list('abcde'))

df1.add(df2,fill_value=0) 对未有值的NaN 以0代替 带入加法

#add #sub #div #mul  加减乘除
#***********************************************
frame=DataFrame(np.random.randn(4,3),columns=list('bde'),index=['Utah','Ohio','Texas','Oregon'])
frame.abs()

#自定义函数
f=lambda x:x.max()-x.min()
#用apply 来执行
frame.apply(f) #对列执行
frame.apply(f,axis=1) #对 行执行

#定义函数
def f(x):
    return Series([x.min(),x.max()],index=['min','nax'])

frame.apply(f)

#************************************************

#格式化 数据
format=lambda x:'%.2f' %x

frame.applymap(format)

#************************************************
#对索引的排序

from pandas import Series,DataFrame
import pandas as pd
import numpy as np

obj=Series(range(4),index=['d','a','b','c'])

obj.sort_index()

frame=DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])

frame.sort_index(axis=1,ascending=False)

frame.sort_index()

frame.sort_index(axis=1)

# 排序

obj=Series([7,-5,7,4,2,0,4])

obj.rank()  #重新 按 升序 从一排序 rank（） 意思 是rank（method=‘average’） 是 又max 和min 两个排名 取均值 得到的。

#另外还有几种排序 平级  处理方式 参见 P140

#******************************************
#索引有可能不是唯一的  带有重复的唯一 可以由以下 来判断

obj.index.is_unique


#***********************************************

df=DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])

df.sum()

df.sum(axis=1)

#axis 轴 行0 列1
#skipna 排除 缺失值

df.mean(axis=1,skipna=False)


#返回出 各种 相关数据
df.describe()

#argmin argmax 能够获取最小值和最大值的索引位置
#idxmin,idxmax 最小值 和 最大值 的索引值

#返回Series 中的 唯一值  重复的 算一次

obj=Series(['a','b','r','j','b'])

uniques=obj.unique()

#排序
uniques.sort()

#计算各值 出现的频率

obj.value_counts()

#处理 缺失值 NA 的方式：

#dropna  丢弃  #fillna  用指定的值 填充缺失数据
#
from numpy import nan as NA

data=Series([1,NA,3.5,NA,7])

data.dropna()
#或者
data[data.notnull()]


data=DataFrame([[1,6.5,3],[1,NA,NA],[NA,NA,NA],[NA,6.5,3]])


#丢弃 全为NA的 行
data.dropna(how='all')


df=DataFrame(np.random.randn(7,3))
df.ix[:4,1]=NA;df.ix[:2,2]=NA

df.dropna(thresh=3)


#填补 缺失 的 数据

#返回新对象
df.fillna(0)
#对现有对象 进行 修改：
_=df.fillna(0,inplace=True)
#********************************************************


from pandas import Series,DataFrame
import pandas as pd
import numpy as np

#层次化 索引

data=Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])

data.index


#选取 子集
data['b']

data['b':'c']

data.ix[['b','d']]


#将Series 划归 成  DataFrame  可以通过unstack方法 重新安排到 一个DataFrame中去

data.unstack()

data.unstack().stack()

#给各层的index 赋名字

frame=DataFrame(np.arange(12).reshape((4,3)),index=[['a','a','b','b'],[1,2,1,2]],columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])

frame.index.names=['key1','key2']
frame.columns.names=['state','color']

frame.swaplevel('key1','key2')


#根据某一列 排序

# level 1 代表  第二列
frame.sortlevel(1)

#统计某个标签下 key2 的和
frame.sum(level='key2')

#set_index 可以将一个或多个列转换 为 行 索引，并创建一个新的 DataFrame:

frame=DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,0,1,2,3]})

frame2=frame.set_index(['c','d'])

frame.set_index(['c','d'],drop=False)

#也可以反过来 用reset_index方法，将提出来的列 重新转移到列中去：

frame2.reset_index()

#面板数据P159 还介绍了一种三维 的面板数据。不常用。

























