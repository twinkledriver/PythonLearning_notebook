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






















