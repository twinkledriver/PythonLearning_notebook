_author_ = 'Administrator'


#coding:utf-8

def run():
    print("hello")


if _name_ == "_main_":
    run()


#from datetime import datetime,date,time
#dt=datetime(2016,3,13,17,14,16)
#w=dt.strftime('%m/%d/%Y %H:%M:%S')
#print w
#q=dt.replace(minute=0,second=0)

#sequence=[1,2,None,4,None,6]
#total=0
#for value in sequence:
#    if value is None:
#        continue  
#    total+=value 
#print total

#def attempt_float(x):
#    try:
#        return float(x)
#    except:
#        return x
#print attempt_float(312)

#float ='%.2f %s is %d'
#k=float %(4.654,'IOIOIO',545)
#print k



    
    #if value == 6:
    #break


#def attempt_float(x):
#    try:
#        return float(x)
#    except (ValueError,TypeError):
#        return x

#print attempt_float((1,6))

#f=open(path,'R')
#try:
#    write_to_file(f)
#finally:
#    f.close()

#k=range(6,50,5)
#print k

#sum=0
#for i in xrange(10):
#    if (i%3==0)&(i%5==0):
#        sum +=i
#        print sum

#x=5
#print 'Non' if x<=0 else 'Yeeee'

#tup=6,7,8
##print tup

#nested_tup=(123,415),(45,78)
#print nested_tup

#tup=('stu',(7,55),[3,0])

#tup[2].append(535)
#c,d,o = tup
#c=c+'tttq'
#print c

#a=(1,2,3,65,4,5,22,2,2)
#k=a.count(2)
#print k

#import pandas
#plot(arange(10))

#tup=('foo','bar','baz')
#b_list=list(tup)

#b_list.append('dwarf')
##print b_list

#c=b_list.pop(2)
#print c
#d='foo' in b_list
#print d

#import bisect
#c=[1,1,2,2,3,3,3,4,5,6,7,8,8,9]

#d=bisect.bisect(c,3)
#print d
#bisect.insort(c,3)
#print c

#some_list=['aaa','bbb','ff']
#mapping=dict((v,i)for i,v in enumerate(some_list))
#print mapping

#print sorted([7,5,6,3,5,47,89])
#print sorted('sdfjheu2 gryysd 2')

#seq1=['df','dgt','345']
#seq2=['dffg','222','bvbrg']
#print zip(seq1,seq2)

#for i,(a,b) in enumerate(zip(s))

#pitchers=[('123','sdfs'),('23546','dsafgdg'),('678','fhu')]
#first,last=zip(*pitchers)
#print first

#a=list(reversed(range(10)))
#print a

##dict={}
#d1={'a':'123123','b':'43636','c':'[sdge,c]'}
##d1[6]='ioioio'
##print d1['c']
##print d1.values()
#d1.update({'123':'ff'})
#print d1

#words=['apple','black','canada','double','egg','atom','book']
#by_letter={}
#for word in words:
#    letter=word[0] 
#    if letter not in by_letter:
#        by_letter[letter]=[word]
#        #print by_letter[letter]
#    else:
#        by_letter[letter].append(word)

#print by_letter[letter]
##print letter

#print hash('string')

#strings=['python','egg','high','green','gen','bin','popmusic']
#c=[x.upper() for x in strings if len(x)>=4]
##print c
#loc_mapping={val:index for index,val in enumerate(strings)}

#loc_mapping2=dict((val,idx)  for val,idx in enumerate(strings))
#print loc_mapping2

#some_tuples=[(1,2,3),(4,5,6,),(7,8,9)]
#flattened=[x for tup in some_tuples for x in tup]
#print flattened

#flattened2=[x for x in tup for tup in some_tuples  ]
##print flattened2

#a=None
#def bind_a_variable():
#    global a
#    a=[]
#bind_a_variable
#print a

#def f():
#    a=5
#    b=6
#    c=7
#    return {'a':a,'b':b,'c':c}
#print f()

#import re
#states=['AdfggFdsf  dg  ','hf894&845ffii  dg','00ghu48EE%#^%','JSgkf']
#def clean(strings):
#    result=[]
#    for value in strings:
#        value=value.strip()
#        value=re.sub('[!#^%?]','',value)
#        value=value.title()
#        result.append(value)
#    return result
#print clean(states)

#import re

#inputStr = "hello crifan, nihao crifan";
#replacedStr = re.sub(r"hello (\w+), nihao \1", "crifanli", inputStr);
#print "replacedStr=",replacedStr; #crifanli


#inputStr = "hello crifan, nihao crifan";
#replacedStr = re.sub(r"hello (\w+), nihao \1", "\g<1>", inputStr);
#print "replacedStr=",replacedStr; #crifan

#def pythonReSubDemo():
 
#    inputStr = "hello 123 world 456";
     
#    def _add111(matched):
#        intStr = matched.group("number"); #123
#        intValue = int(intStr);
#        addedValue = intValue + 111; #234
#        addedValueStr = str(addedValue);
#        return addedValueStr;
         
#    replacedStr = re.sub("(?P<number>\d+)", _add111, inputStr,1);
#    print "replacedStr=",replacedStr; #hello 234 world 567
#pythonReSubDemo()


#import re
#states=['AdfggFdsf  dg  ','hf894&845ffii  dg','00ghu48EE%#^%','JSgkf']
#def remove(value):
#    return re.sub('[!@#$%^&*()]','',value)

#clean_ops=[str.strip,remove,str.title]

#def clean_strings(strings,ops):
#    result=[]
#    for value in strings:
#        for function in ops:
#            value=function(value)
#        result.append(value)
#    return result

#print clean_strings(states,clean_ops)

#def apply(list,f):
#    return [f(x) for x in list]

#ints=[4,5,6,9,8,7,1,3]
#print apply(ints,lambda x: x*2)

#strings=['asdg','dgur','fsghdefiu','opoo','qwr']

#strings.sort(key=lambda x:len(set(list(x))))

#print strings

#def make_closure(a):
#    def closure():
#        print('I know the secret:%d'  %a)
#    return closure

#make_closure(7)

#def squares(n=10):
#    t=(n**2)
#    for i in xrange(1,n+1):
#        print 'Generate squares from 1 to %d'%t
#        yield i**2
#gen=squares()

#for x in gen:
#    print x,

#def make_change(amount,coins=[1,5,10,25],hand=None):
#    hand=[]if hand is None else hand
#    if amount==0:
#        yield hand
#    for coin in coins:
#        if coin>amount or (len(hand)>0 and hand[-1]<coin ):
#            continue
#        for result in make_change(amount-coin,coins=coins,hand=hand+[coin]):
#            yield result

#for way in make_change(100,coins=[5,10,25,50]):
#    print way


#def make_change(amount,coins=[1,5,10,50],hand=None):
#    hand=[]if hand is None else hand
#    if amount==0:
#        yield hand
#    for coin in coins:
#        if coin>amount or (len(hand)>0 and hand[-1]<coin ):
#            continue
#        for result in make_change(amount-coin,coins=coins,hand=hand+[coin]):
#            yield result

#for way in make_change(15,coins=[5]):
#    print way



#import itertools
#first_letter=lambda x:x[0]
#names=['apple','away','candy','disc','dos','fully']
#for letter,names in itertools.groupby(names,first_letter):
#    print letter,list(names)



#n=15
#s=range(1,n)
#c=0;

#print sum(map(lambda x:int(x),unicode(134)))

#[json.loads(line) for line in open(path)]





# import json
# path='C:\Users\Administrator\Desktop\usagov_bitly_data2012-03-16-1331923249.txt'
# records=[json.loads(line) for line in open(path)]
#
# def top_counts(count_dict,n=5):
#     value_key_pairs=[(count,tz) for tz,count in count_dict.items()]
#     value_key_pairs.sort()
#     return value_key_pairs[-n:]
#
# counts=get_counts(time_zones)

#def get_counts(sequence):
#    counts={}
#    for x in sequence:
#        if x in counts:
#            counts[x]+=1
#        else:
#            counts[x]=1
#        return counts

#    time_zones=[rec['tz'] for rec in records if 'tz' in rec] 

#    from collections import Counter
#    counts=Counter(time_zones)
#    counts.most_common(10)  //????????? ???????? ????? ????????????????
#
# from pandas import DataFrame,Series
# import pandas as pd;import numpy as np
# frame=DataFrame(records)
# frame
#
#
# tz_counts=frame['tz'].value_counts()
# #**************************************************
# p26
# clean_tz=frame['tz'].fillna('Missing')
# clean_tz[clean_tz=='']='Unknown'
# tz_counts=clean_tz.value_counts()
#
# tz_counts[:10]
#
#
#
#
# tz_counts[:10].plot(kind='barh',rot=0)
#
#
# frame['a'][50]
#
#
# results=Series([x.split()[0] for x in frame.a.dropna()])
#
# results.value_counts()[:8]
#
# cframe=frame[frame.a.notnull()]
#
# operating_system=np.where(cframe['a'].str.contains('Windows'),'Windows','Not Widows')
#
# operating_system[:5]
#
# by_tz_os=cframe.groupby(['tz',operating_system])
#
# agg_counts=by_tz_os.size().unstack().fillna(0)
#
# agg_counts[:10]
#
# indexer=agg_counts.sum(1).argsort()
#
# indexer[:10]
#
# count_subset=agg_counts.take(indexer)[-10:]
#
# count_subset.plot(kind='barh',stacked=True)
#
# #//????????????? ??????????
#
# normed_subset=count_subset.div(count_subset.sum(1),axis=0)
#
# normed_subset.plot(kind='barh',stacked=True)

#******************************************************
#P30

import pandas as pd
unames=['user_id','gender','age','occupation','zip']


users=pd.read_table('C:\Users\Administrator\Desktop\movielens\users.dat',sep='::',header=None,names=unames)

users[:5]

rnames=['user_id','movie_id','rating','timestamp']
ratings=pd.read_table(r'C:\Users\Administrator\Desktop\movielens\ratings.dat',sep='::',header=None,names=rnames)        #录文件前加'r'，跟前面不一样，否则报错 这个书上也不一样。


mnames=['movie_id','title','genres']
movies=pd.read_table('C:\Users\Administrator\Desktop\movielens\movies.dat',sep='::',header=None,names=mnames)


#movies=pd.read_table('C:\Users\Administrator\Desktop\movielens\README')

#*******************************
#P31
users[:5]
ratings[:5]
movies[:5]

#p32
#合并用户、得分、电影

data=pd.merge(pd.merge(ratings,users),movies)

data.ix[0]

mean_ratings=data.pivot_table('rating',columns=['gender'],aggfunc='mean')
#有错不支持rows了 没有定义 而 col也要写成columns rows改成index
mean_ratings = data.pivot_table('rating',index='title',columns='gender',aggfunc = 'mean')

mean_ratings[:5]


#补充练习：http://python.jobbole.com/81212/   关于pivot_table的使用


import pandas as pd
import numpy as np


#这里需要安装打开excel文件的组件：xlrd（官方下载） 安装用Windows下的cmd以命令行来安装
df = pd.read_excel("C:\Users\Administrator\Desktop\sales-funnel.xlsx")
df.head()

df["Status"] = df["Status"].astype("category")
df["Status"].cat.set_categories(["won","pending","presented","declined"],inplace=True)

pd.pivot_table(df,index=["Name"])

pd.pivot_table(df,index=["Name","Rep","Manager"])

pd.pivot_table(df,index=["Manager","Rep"])

#它已经开始通过将“Rep”列和“Manager”列进行对应分组，来实现数据聚合和总结

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"])


#“Price”列会自动计算数据的平均值，但是我们也可以对该列元素进行计数或求和。要添加这些功能，使用aggfunc和np.sum就很容易实现。
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=np.sum)

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],
               columns=["Product"],aggfunc=[np.sum])

pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],
               columns=["Product"],aggfunc=[np.sum],fill_value=0)

pd.pivot_table(df,index=["Manager","Rep"],values=["Price","Quantity"],
               columns=["Product"],aggfunc=[np.sum],fill_value=0)

pd.pivot_table(df,index=["Manager","Rep","Product"],
               values=["Price","Quantity"],aggfunc=[np.sum],fill_value=0)

pd.pivot_table(df,index=["Manager","Rep","Product"],
               values=["Price","Quantity"],
               aggfunc=[np.sum,np.mean],fill_value=0,margins=True)


pd.pivot_table(df,index=["Manager","Status"],values=["Price"],
               aggfunc=[np.sum],fill_value=0,margins=True)

pd.pivot_table(df,index=["Manager","Status"],columns=["Product"],values=["Quantity","Price"],
               aggfunc={"Quantity":len,"Price":np.sum},fill_value=0)
#***************************
P33

ratings_by_title=data.groupby('title').size();

ratings_by_title[:10]

active_titles=ratings_by_title.index[ratings_by_title>=250]
active_titles

mean_ratings=mean_ratings.ix[active_titles]

mean_ratings

top_female_ratings=mean_ratings.sort_index(by='F',ascending=False)

top_female_ratings[:10]

#*************************************
P34
mean_ratings['diff']=mean_ratings['M']-mean_ratings['F']
sorted_by_diff=mean_ratings.sort_index(by='diff')
sorted_by_diff[:15]
sorted_by_diff[::-1][:15]
#*************************************
P35

#标准差

rating_std_by_title=data.groupby('title')['rating'].std()

rating_std_by_title=rating_std_by_title.ix[active_titles]

rating_std_by_title.order(ascending=False)[:10]

#电影评分分析 结束
#*********************
P37
#婴儿姓名分析


#暂时只能放桌面读取，有文件夹就报错,不知为何？
#文件名每个子目录必须大写首字母
import pandas as pd
names1880=pd.read_csv('C:\Users\Administrator\Desktop\yob1880.txt',names=['names','sex','births'])

names1880.groupby('sex').births.sum();




#定义年份，为了标记
import pandas as pd
years=range(1880,2011);
pieces=[];
columns=['name','sex','births']

#注意缩进 代表循环 内外部
for year in years:
    path='C:\Users\Administrator\Desktop\Names\yob%d.txt'%year
    frame=pd.read_csv(path,names=columns)
    frame['year']=year
    pieces.append(frame)

names=pd.concat(pieces,ignore_index=True)

names

total_births=names.pivot_table('births',index=['year'],columns=['sex'],aggfunc='sum')
#只列出尾几行数据
total_births.tail()

#画图

total_births.plot(title='Total births by sex and year')


#比例化
def add_prop(group):
    births=group.births.astype(float)

    group['prop']=births/births.sum()
    return group

names=names.groupby(['year','sex']).apply(add_prop)

names

#验证是否归一

np.allclose(names.groupby(['year','sex']).prop.sum(),1)


#前1000个
def get_top1000(group):
    return group.sort_index(by='births',ascending=False)[:1000]

grouped=names.groupby(['year','sex'])
top1000=grouped.apply(get_top1000)

#另一种
pieces_another=[]
for year,group in names.groupby(['year','sex']):
    pieces_another.append(group.sort_index(by='births',ascending=False)[:1000])

top1000_another=pd.concat(pieces_another,ignore_index=True)

#************************************
#分析命名趋势
#P40

boys=top1000[top1000.sex=='M']

girls=top1000[top1000.sex=='F']

total_births=top1000.pivot_table('births',index='year',columns='name',aggfunc='sum')

#子集
subset = total_births[['John','Harry','Mary','Marilyn']]
#大小写一定要对，否则不出图。
subset.plot(subplots=True,figsize=(12,10),grid=False,title="Number of births per year")


#前1000个名字 总和 人数占 的比例
table=top1000.pivot_table('prop',index='year',columns='sex',aggfunc=sum)


#yticks=np.linspace(a,b,c) 起始点：a 结束点：b 一共作c个点。 xticks=range（a,b,c） 起始点：a 结束点：b  c:间隔
table.plot(title='Sum of table1000.prop by year and sex',yticks=np.linspace(0,1.2,13),xticks=range(1880,2020,10))


#*********************************************************
#P42
#统计2010年 boys 名字比例 累加 并判断大于50% 时 的名字累积数量
df=boys[boys.year==2010]

prop_cumsum=df


prop_cumsum=df.sort_index(by='prop',ascending=False).prop.cumsum()

prop_cumsum[:10]

prop_cumsum.searchsorted(0.5)


#1900年

df=boys[boys.year==1900]
in1900=df.sort_index(by='prop',ascending=False).prop.cumsum()

in1900.searchsorted(0.5)+1

#结论说明 名字更加多样了

def get_quantile_count(group,q=0.5):
    group=group.sort_index(by='prop',ascending=False)
    return group.prop.cumsum.searchsorted(q)+1

diversity=top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity=diversity.unstack('sex')

#**************************************
#P44 最后一个字母的统计

get_last_letter=lambda x:x[-1]
last_letters=names.name.map(get_last_letter)
get_last_letter.name='last_letter'

table=names.pivot_table('births',index=last_letters,columns=['sex','year'],aggfunc=sum)

subtable=table.reindex(columns=[1910,1960,2010],level='year')

subtable.head()

subtable.sum()

#条形图绘制

letter_prop=subtable/subtable.sum().astype(float)

import matplotlib.pyplot as plt

fig,axes=plt.subplots(2,1,figsize=(10,8))
letter_prop['M'].plot(kind='bar',rot=0,ax=axes[0],title='Male')
letter_prop['F'].plot(kind='bar',rot=0,ax=axes[1],title='Female')

letter_prop=table/table.sum().astype(float)
dny_ts=letter_prop.ix[['d','n','y'],'M'].T

dny_ts.head()
dny_ts.plot()

#****************************
#P46


#提取部分包含字，掩码
import numpy as np

all_names=top1000.name.unique()

mask=np.array(['lesl' in x.lower() for x in all_names])

lesley_like=all_names[mask]

lesley_like

#处理_过滤

filtered =top1000[top1000.name.isin(lesley_like)]

filtered.groupby('name').births.sum()


#处理——性别 年代 聚合

table=filtered.pivot_table('births',index='year',columns='sex',aggfunc='sum')

table=table.div(table.sum(1),axis=0)
table.tail()

table.plot(style={'M':'k-','F':'k--'})


#******************************************************#******************************************************
#******************************************************#******************************************************
#******************************************************#******************************************************
#第三章 IPython

import numpy as np

data={i:randn() for i in range(7)}

#Pylab具有 Tab 补全功能
#对象.<Tab> 展示可以用的方法

#%timeit + 函数 可以检测语句的执行时间。

#查看魔术命令
#%magic


#打印输入过的命令行
#%hist

#调用Qt框架GUI控制台

#ipython qtconsole --pylab=inline

#$ ipython -- pylab

# _  和  _ _         可以调用出最近执行函数的结果
    #_i+行号  对应行号输入的变量
    #_+ 行号  对应行号输出的变量

# 记录控制台会话，日志
#%logstart
#%logon

#以！开头的命令 表明要再shell中执行。

#%debug 迅速调试

#%pdb 开启%debug

#单步调试 查看书P67页

#******************************************
#P77
#reload(模块) 可以加载最新的模块


#*****************************************#*****************************************
#*****************************************#*****************************************
#*****************************************#*****************************************

#第四章 数组与矢量计算
#P83

#ndarray 构造多维数组对象，所有元素必须是相同类型。
import numpy as np

data=array([[0.9264,-0.5654,0.4548],[0.5498,0.654,0.454]])

data*10

data.shape #返回各维度大小的元组
data.dtype #返回数据类型


#创建数组
data1=[6,7.5,8,0,1]
arr1=np.array(data1)

data2=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(data2)

#zeros 可以创建一个全0的数组
#ones  可以创建一个全1的数组
#empty 可以创建一个 没有任何具体值的数组(并非0)

np.zeros(10)
np.zeros((10,10))

np.empty((2,3,2))
#0到9
np.arange(10)

#未说明 都是浮点数

eye(5)


#*********************************************

import numpy as np

arr=np.array([1,2,3,4,5])

arr.dtype


#转换数据类型  astype
float_arr=arr.astype(np.float64)

#转换字符串 到数值 类型一定要合适，否则报错
numric_strings=np.array(['1.25','5.36','-53.1'])
numric_strings.dtype
numric_strings.astype(float)

#不同大小的数组之间的运算叫做广播 后面讲解

#切片

arr=np.arange(10)

arr[5:8] #截出5到8位

arr[5:8]=19 #替换

 #此种操作 直接影响原始 数据


 #slice

arr_slice=arr[5:8]

arr_slice[1]

#要想不改变源数据，而是 用副本的形式

arr[5:8].copy()


arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
#递归访问

#以下两种方式 等价

arr2d[0,2]
arr2d[0][2]

#***********************************************
import numpy as np

arr2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[1:3] # 显示3个 从第一组之后开始  [[4,5,6],[7,8,9]]

arr2d[:2,1:]# 显示 前两组 每个从第一维之后开始 [[2,3],[5,6]]

arr2d[1,:2] #第二个（数组 0代表第1个），只显示前两个 [4.5]

arr2d[2,:1]# 同上         [7]

arr2d[:,:1] #全部考虑 只显示 第一维 [1],[4],[7]

names=np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])

data=randn(7,4) #7组 每组4维

names=='Bob'

#以下这些用法都是 上面知识点的混合使用

data[names=='Bob']

data[names=='Bob',2:]


#Python 中的与 只有一个&  或也只有一个| 来表示

#花式索引

arr=np.empty((8,4))
for i in range(8):
    arr[i]=i

arr[[4,2,6,7]]

#关于reshape 的第一次使用

arr=np.arange(32).reshape(8,4) # -- 8 行四列 从 0到 31

#花式索引 是复制元素 并不改变本身的值


#转置操作 T

arr=np.arange(15).reshape(3,5)

arr.T

#计算内积
np.dot(arr.T,arr)

arr=np.arange(16).reshape(2,2,4)


#没看懂 怎么变换的
arr.transpose((1,0,2))

#可以采纳
arr.swapaxes(1,2)

#*********************************************************************
import numpy as np

#ufunc
#sqrt 开方运算

arr=np.arange(10)

np.sqrt(arr)

#x=randn(8)
#y=randn(8)
np.maximum(x,y)

#modf()函数 返回 数组的整数和小数部分
np.modf(x)

#书P99

#abs 返回绝对值
#square 平方
#sign 正负号
#ceil 大于等于该值的最小整数
#floor 小于等于该值的最大整数
#rint  四舍五入

#add 对应元素相加
#subtract  前面减去后面的对应元素
#multiply 乘法（元素）
#power 乘方
#copysign 复制符号
#logical_add logical_or logical_xor


points=np.arange(-5,5,0.01)
xs,ys=np.meshgrid(points,points)


from matplotlib import mpl

import numpy as np



#********************************************绘图函数 比较重要
import numpy as np
import matplotlib.pyplot as plt


z=np.sqrt((xs**2+ys**2)) #平方和 再开方

#绘图
plt.imshow(z,cmap=plt.cm.gray);  #灰度图


plt.colorbar()#灰度标尺

#$\sqrt{x^2+y^2}$ 以符号形式展现

plt.title("Image plot of $\sqrt{x^2+y^2}$ for a grid of values")



#where的用法 类似 语法 x if condition else y

xarr=np.array([1.1,1.2,1.3,1.4,1.5])
yarr=np.array([2.1,2.2,2.3,2.4,2.5])
cond=np.array([True,False,True,True,False])

result=[(x if c else y)
        for x,y,c in zip(xarr,yarr,cond)]  #zip 配对 对应位置

#而用where 函数去实现 更为简单

result=np.where(cond,xarr,yarr)

#比如实现 值的替换

arr=randn(4,4)

np.where(arr>0,2,-3)


#更为复杂的语法 也能巧妙的实现

result=[]
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)

#等价于

np.where(cond1&cond2,0,np.where(cond1,1,np.where(cond2,2,3)))



arr=np.random.randn(5,4)
arr.mean()

np.mean(arr)

#可以只对x轴操作
arr.mean(axis=1)
arr.sum(0)

#累加  累乘 函数

arr=np.array([[0,1,2],[3,4,5],[6,7,8]])

arr.cumsum(0)
arr.cumprod(1)  #0和1 代表维度


#布尔值会被强制转换成1（True）和0（False）

#sort 函数可以排序

arr=randn(8)

arr.sort()

#对单一维度 排序

arr=randn(5,3)

arr.sort(0)
#返回特定位置的值 如分位数

large_arr=randn(1000)
large_arr.sort()
large_arr[int(0.05*len(large_arr))]

#返回数组中的唯一化后的值 用unique()  除去 重复值

names=np.array(['Bob','Joe','Will','Bob','Joe','Joe'])

np.unique(names)

#P107 还有其他集合运算



#保存数组 以二进制 形式 到磁盘

arr=np.arange(10)

np.save('some_array',arr)

np.load('some_array.npy')

np.savez('array_archive.npz',a=arr,b=arr)

arch=np.load('array_archive.npz')

mean_ratings

arr=np.loadtxt('example.txt',delimiter=',')


































































































