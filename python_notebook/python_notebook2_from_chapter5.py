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

#**********************************************************************#**********************************************************************
#**********************************************************************#**********************************************************************
#**********************************************************************#**********************************************************************
#第六章  数据加载、存储与文件格式

#最常用的解析函数：

read_csv:#默认分隔符 为逗号
read_table:#默认分割符 为"\t"
read_clipboard:#读取 粘贴板 中的内容

import pandas as pd

from pandas import Series,DataFrame

df=pd.read_csv('ch06/ex1.csv')

pd.read_table('ch06/ex1.csv',sep=',')

#自己 指定 列名
pd.read_csv('ch06/ex2.csv',header=None)

pd.read_csv('ch06/ex2.csv',names=['a','b','c','d','Message'])

names=['a','b','c','d','Message']
pd.read_csv('ch06/ex2.csv',names=names,index_col='Message')

parsed=pd.read_csv('ch06/csv_mindex.csv',index_col=['key1','key2'])

#以上这些 都有明确 的分隔符 容易 载入 成表

#下面要介绍自己编写正则表达式 来作为 分隔符

list(open('ch06/ex3.txt'))

#用正则表达式 \s+ 整理 空格
result=pd.read_csv('ch06/ex3.txt',sep='\s+')


#用skiprows 可以跳出文件 的某些行
pd.read_csv('ch06/ex4.csv',skiprows=[0,2,3])

result=pd.read_csv('ch06/ex5.csv')

result=pd.read_csv('ch06/ex5.csv',na_values=['NULL'])

#P167 描述了 很挫read_csv函数 的参数

#读取文件 中 部分数据

result=pd.read_csv('ch06/ex6.csv')


result=pd.read_csv('ch06/ex6.csv',nrows=5)

#逐块读取文件

chunker=pd.read_csv('ch06/ex6.csv',chunksize=1000)

tot=Series([])

for piece in chunker:
    tot=tot.add(piece['key'].value_counts(),fill_value=0)

#提取key 列 按计数顺序 排序
tot=tot.order(ascending=False)

tot[:10]

#******************************************************
#写出 文本 格式

#读入
data=pd.read_csv('ch06/ex5.csv')

#写出 数据 以逗号分开
data.to_csv('out.csv')

#加 指定 分隔符'|',  并未 输出 实际文件  而是 打印
data.to_csv(sys.stdout,sep='|')
#空值  填  NULL

data.to_csv(sys.stdout,sep='|',na_rep='NULL')

#禁用 行和列的 标签

data.to_csv(sys.stdout,sep='|',na_rep='NULL',index=False,header=False)

import  csv
f=open('ch06/ex7.csv')
reader=csv.reader(f)

for line in reader:
    print line

lines=list(csv.reader(open('ch06/ex7.csv')))
header,values=lines[0],lines[1:]

zip #配对

data_dict={h:v for h,v in zip(header,zip(*values))}  #注意 这里 的  *values

#****************************************************

#定义一个类  对常规文本 参考书 P172
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'

with open('data.csv','w') as f:
    writer=csv.writer(f,dialect=my_dialect)
    writer.writerow(('One','Two','Three'))
    writer.writerow(('A','B','C'))

#****************************************************

#JSON格式: JavaScript Object Notation  的简写： HTTP 请求Web浏览器 和其他应用程序 之间发送数据的标准格式

obj="""
{"name":"Wes",
 "places_lived":["United States","Spain","Germany"],
 "pet":null,
 "sibling":[{"name":"Scott","age":25,"pet":"zuko"},
            {"name":"Katie","age":33,"pet":"Cisco"}]
}
"""

#转换成Python格式
import json
result=json.loads(obj)

#转换成JSon 格式
asjson=json.dumps(result)

#提取部分 以DataFrame形式 展现
siblings=DataFrame(result['sibling'],columns=['name','age'])
#**********************************************
#对HTML 和 XML 的提炼 和 处理 P174

#加入 调用的 库 （下载 安装）
from lxml.html import parse
from urllib2 import  urlopen

#提炼网页对象
parsed=parse(urlopen('http://blog.sciencenet.cn/home.php?mod=space&uid=425437'))


doc=parsed.getroot()
#提取  HTML （链接 是 a标签）
links=doc.findall('.//a')
# 随便选取一个
lnk=links[2]
# 获取 超链接引用
lnk.get('href')

lnk.text_content()
#所有链接
urls=[lnk.get('href') for lnk in doc.findall('.//a')]


#打印所有URL 到 out_href 文件中
f=open('out_href.csv','w')
for i in urls:
    k=' '.join([str(j) for j in i])
    f.write(k+"\n")
f.close()
urls.to_csv('out_href.csv')


#书上的例子 P174

from lxml.html import parse
from urllib2 import urlopen

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))

doc = parsed.getroot()

tables=doc.findall('.//table')

calls=tables[1]
puts=tables[2]

#标题行 th 单元 里    数据行 td 行 里
rows=calls.findall('.//tr')

def _unpack(row,kind='td'):
    elts=rows.findall('.//%s'%kind)
    return [val.text_content() for val in elts]

_unpack(rows[0],kind='th')

#**************************************************
#P176 整体 程序

from pandas.io.parsers import TextParser

def parse_options_data(table):
    rows=table.findall('.//tr')
    header=_unpack(rows[0],kind='th')
    data=[_unpack(r) for r in rows[1:]]
    return TextParser(data,names=header).get_chunk()

#*********************************************************
#XML 文件的解析 (Extensible Markup Language)

#读取、
from lxml import objectify
path='nyct_ene.xml'
parsed=objectify.parse(open(path))
root=parsed.getroot()

data=[]

skip_fields=['responsecode']

for elt in root.outage:
    el_data={}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag]=child.pyval
    data.append(el_data)

from pandas import Series,DataFrame

perf=DataFrame(data)

#*****************************************************

from StringIO import StringIO
tag='<a href="http://www.google.com">Google</a>'

root=objectify.parse(StringIO(tag)).getroot()

#********************************************************
#以二进制 数据格式 存储

import pandas as pd
frame=pd.read_csv('ch06/ex1.csv')

frame.to_pickle('ch06/frame_pickle')


#****************************************************
#HDF5 （hierarchical data format） 层次型 数据格式
#用于 高效读取 二进制存储 的数据格式
#有两种接口 PyTables 和h5py

store=pd.HDFStore('mydata.h5')
store['obj1']=frame
store['obj1_col']=frame['a']

#*******************************************
#读取 excel 文件

xls_file=pd.ExcelFile('data.xls')
table=xls_file.parse('Sheet1')

#******************************************
#分析 HTML JSON格式 数据  用其提供的API 接口 P181

import requests
url='http://live.qq.com/json/movie/all/hot2/list_7.json'

resp=requests.get(url)

resp

import json
data=json.loads(resp.text)

data.keys()

#************************************************
#与数据库的 交互
#yong pandas 提供 的 嵌入式 SQLite 数据库

import  sqlite3
query="""
CREATE TABLE test
(a VARCHAR(20),b VARCHAR (20),
c REAL  ,d INTEGER );"""
con=sqlite3.connect(':memory:')
con.execute(query)
con.commit()


#插入数据
data=[('Atlanta','Georgia',1.25,6),
      ('Tallahassee','Florida',2.6,3),
      ('Sacramento','Califonia',1.7,5)]
stmt="INSERT INTO test VALUES(?,?,?,?)"

con.executemany(stmt,data)
con.commit()

#查看数据
cursor=con.execute('select * from test')  #地址
rows=cursor.fetchall()

#至于fetchall()则是接收全部的返回结果行 row就是在python中定义的一个变量，
# 用来接收返回结果行的每行数据。同样后面的r也是一个变量，用来接收row中的每个字符，
# 如果写成C的形式就更好理解了
#for(string row = ''; row<= cursor.fetchall(): row++)
#    for(char r = ''; r<= row; r++)
#printf("%c", r);
#大致就是这么个意思！

#传递给DataFrame,并用cursor 游标 添加 columns  “A”“B”“C”“D”。。。



cursor.description

from pandas import DataFrame
DataFrame(rows,columns=zip(*cursor.description)[0])

#另外一种简单的方式  read_frame函数

import pandas.io.sql as sql

sql.read_frame('select * from test',con)

#***********************************************#*********************************************
#***********************************************#*********************************************
#***********************************************#*********************************************
#第七章 数据规整化：清理 转换 合并 重塑 P186


# 合并 操作  合并 DataFrame

from pandas import DataFrame
import pandas as pd

df1=DataFrame({'key':['b','b','a','c','a','a','b'],
               'data1':range(7)})
df2=DataFrame({'key':['a','b','d','c'],
               'data2':range(4)})


#merger 只保留 df1,df2 共有的key 值

pd.merge(df1,df2)

#更为 规范的 写法  merge 后指定  键  为 重叠列名

pd.merge(df1,df2,on='key')

# 若 键 值 名 不同，也可以 分别指定 各自的 键值 名

df3=DataFrame({'lkey':['b','b','a','c','a','a','b'],
               'data1':range(7)})
df4=DataFrame({'rkey':['a','b','d'],
               'data2':range(3)})

pd.merge(df3,df4,left_on='lkey',right_on='rkey')

pd.merge(df3,df4,left_on='lkey',right_on='rkey',how='outer')

# d多对多  是 以 笛卡尔积  左右 相乘 个元素

df1=DataFrame({'key':['b','b','a','c','a','b'],
               'data1':range(6)})

df2=DataFrame({'key':['a','b','a','b','d'],
               'data2':range(5)})

pd.merge(df1,df2,how='inner')

#多个键 合并

left=DataFrame({'key1':['foo','foo','bar'],
                'key2':['one','two','one'],
                'lval':[1,2,3]})

right=DataFrame({'key1':['foo','foo','bar','bar'],
                'key2':['one','one','one','two'],
                'rval':[4,5,6,7]})

pd.merge(left,right,on=['key1','key2'],how='outer')

#*******************************************************
#索引层面上 的合并 P191

from pandas import DataFrame
import pandas as pd
left1=DataFrame({'key':['a','b','a','a','b','c'],
                 'value':range(6)})

right1=DataFrame({'group_val':[3.5,7]},index=['a','b'])

pd.merge(left1,right1,left_on='key',right_index=True)

pd.merge(left1,right1,left_on='key',right_index=True,how='outer')

#******************************************
lefth=DataFrame({'key1':['Ohio','Ohio','Ohio','Nevada','Nevada'],
                 'key2':[2000,2001,2002,2001,2002],
                 'data':np.arange(5)})

righth=DataFrame(np.arange(12).reshape((6,2)),
                 index=[['Nevada','Nevada','Ohio','Ohio','Ohio','Ohio'],[2001,2000,2000,2000,2001,2002]],
                 columns=['event1','event2'])


#left_on 以key1 key2 为标杆 event 项 没有的填NaN  另外 默认是 交集
pd.merge(lefth,righth,left_on=['key1','key2'],right_index=True)

pd.merge(lefth,righth,left_on=['key1','key2'],right_index=True,how='outer')

#***********************************************
left2=DataFrame([[1,2],[3,4],[5,6]],index=['a','c','e'],columns=['Ohio','Nevada'])

right2=DataFrame([[7,8],[9,10],[11,12],[13,14]],index=['b','c','d','e'],columns=['Missouri','Alabama'])

pd.merge(left2,right2,how='outer',left_index=True,right_index=True)

#或者 效果一样
left2.join(right2,how='outer')

another=DataFrame([[7,8],[9,10],[11,12],[16,17]],index=['a','c','e','f'],columns=['New York','Oregon'])

[right2,another]

left2.join([right2,another])

left2.join([right2,another],how='outer')

#***********************************************************************
#轴向连接 也被称为 连接 concatenation  绑定 binding  堆叠  stacking

#其中 包含一个 concatenation()函数

arr=np.arange(12).reshape((3,4))

np.concatenate([arr,arr],axis=1)
np.concatenate([arr,arr],axis=0)

from pandas import Series,DataFrame

s1=Series([0,1],index=['a','b'])

s2=Series([2,3,4],index=['c','d','e'])

s3=Series([5,6],index=['f','g'])

s4=pd.concat([s1* 5,s3])


# 默认concat 是axis=0  的序列
pd.concat([s1,s2,s3])
# 如果 改成axis=1 就会变成 一个 DateFrame
pd.concat([s1,s2,s3],axis=1)

s4=pd.concat([s1* 5,s3])



pd.concat([s1,s4])

pd.concat([s1,s4],axis=1,join_axes=[['a','c','b','e']])

result=pd.concat([s1,s1,s3],keys=['one','two','three'])

#解除笛卡尔对应形式，一一对应。
result.unstack()

#**************************************************

df1=DataFrame(np.arange(6).reshape(3,2),index=['a','b','c'],columns=['one','two'])

df2=DataFrame(5+np.arange(4).reshape(2,2),index=['a','c'],columns=['three','four'])

pd.concat([df1,df2],axis=1,keys=['level1','level2'])

pd.concat([df1,df2],axis=1,keys=['level1','level2'],names=['upper','lower'])

#concat 函数 相关应用P198
#**************************************************

#合并数据

a=Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index=['f','e','d','c','b','a'])

b=Series(np.arange(len(a),dtype=np.float64),index=['f','e','d','c','b','a'])

#where  if  - else  结构
np.where(pd.isnull(a),b,a)

#******************************************
#combine_first 缺失的数据 由combine 的对象来补齐

#stack 列 变 行
#unstack  行 变 列

data=DataFrame(np.arange(6).reshape((2,3)),index=pd.Index(['Ohio','Colorado'],name='state'),columns=pd.Index(['one','two','three'],name='number'))
#变成一个Series
result=data.stack()

result.unstack()

#也可以 单独 解放一个 列的 编号

result.unstack('state')

#*************************************************
#格式转换 长格式 转换成 短格式

#载入 整个数据
data = pd.read_csv('ch07/macrodata.csv')

#提取数据中  的 年份  月份 作为 时间序列化 后IU面 要用  并命名为data
periods = pd.PeriodIndex(year=data.year, quarter=data.quarter, name='date')

#做一个DataFrame的表 数据用csv中的 提取 realgdp infl unemp 3个字段 来分析 而
#列标 用的是 上面提取 的时间序列 格式

data = DataFrame(data.to_records(),
                 columns=pd.Index(['realgdp', 'infl', 'unemp'], name='item'),
                 index=periods.to_timestamp('D', 'end'))

#data unstack 化  可以一步一步 执行 观察 变动
ldata = data.stack().reset_index().rename(columns={0: 'value'})

ldata[:10]
#pivot 可以将数据库 格式 的 数据 转换成 DataFrame

pivoted = ldata.pivot('date', 'item', 'value')

pivoted.head()

#pivot 是一种快捷的方式 等价于
unstacked=ldata.set_index(['date','item']).unstack('item')

#************************************
#在知道 数据怎么重排之后  看 数据 怎么转换

#过滤

data=DataFrame({'k1':['one']*3+['two']*4,'k2':[1,1,2,3,3,4,4]}

#返回 Series 是否重复 行
data.duplicated()
#移出 重复 行
data.drop_duplicates()
#也可以 指定 列 移出
data.drop_duplicates(['k1'])

data['v1']=range(7)


#************************************

#映射

data=DataFrame({'food':['bacon','pulled pork','bacon','Pastrami','corned beef','Bacon','pastrami','honey ham','nova lox'],'ounces':[4,3,12,6,7.5,8,3,5,6]})

meat_to_animal={'bacon':'pig','pulled pork':'pig','pastrami':'cow','corned beef':'cow','honey ham':'pig','nova lox':'salmon'}

#让二者 对应起来 这里存在 一个大小写的问题

data['animal']=data['food'].map(str.lower).map(meat_to_animal)
#也可以 用 lambda 函数 的 方式

data['food'].map(lambda x:meat_to_animal[x.lower()])


#替换
from pandas import  Series

data=Series([1,-999,2,-999,-1000,3])

#替换 函数
data.replace(-999,np.nan)

#替换多值
data.replace([-999,-1000],np.nan)

#多值对应替换

data.replace([-999,-1000],[np.nan,0])

#也可以用 字典的 方式 对应替换
data.replace({-999:np.nan,-1000:0})

#******************************************************
#重命名 轴索引

data=DataFrame(np.arange(12).reshape((3,4)),index=['Ohio','Colorado','New York'],columns=['one','tow','three','four'])

data.index=data.index.map(str.upper)
#并不修改原始 数据 用方法rename

data.rename(index=str.title,columns=str.upper)

# 可以修改 列名 或者 行名  替换的方式  原始 并不改变

data.rename(index={'OHIO':'INDIANA'},columns={'three':'peekaboo'})

_=data.rename(index={'OHIO':'INDIANA'},inplace=True)

#离散化 和面元  分析

ages=[20,22,25,27,21,23,37,31,61,45,64,32]

#划分年龄段

bins=[18,25,35,60,100]

#使用cut  函数
cats=pd.cut(ages,bins)

#每个数据 都划归内 哪一类

#归为 第几个 分组 中
cats.labels
#分了 哪些 组
cats.levels

#统计 分段
pd.value_counts(cats)

#right=False 可以 改变 开闭区间

pd.cut(ages,[18,26,36,61,100],right=False)

#可以 给 自己 的分组 取 名字

group_names=['Youth','YoungAdult','MiddleAge','Senior']

pd2=pd.cut(ages,bins,labels=group_names)
pd2.value_counts()

#若不给出分组 而是以参数 替代 函数会自动 根据 比例 分组

data=np.random.randn(20)

#precision 小数后 几位
pd.cut(data,4,precision=3)

#根据样本分位数 进行切割

import numpy as np

data=np.random.randn(1000)

cats=pd.qcut(data,4)

pd.value_counts(cats)

#自定义 分位数

pd.qcut(data,[0,0.1,0.5,0.9,1])

#检测和过滤 异常值

#异常值 outlier  孤立点

import numpy as np
import pandas as pd
from pandas import DataFrame,Series

np.random.seed(12345)

#seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，
# 则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，
# 此时每次生成的随机数因时间差异而不同。

#seed 是为了记录后面要生成 的随机数，不然 随机数 会一直变

data=DataFrame(np.random.randn(1000,4))

#找出某列中 绝对值大于3 的行

col=data[3]
col[np.abs(col)>3]

#绝对值 大于3 的所有行  注意 any(1)的用法

data[(np.abs(data)>3).any(1)]

#下面的 函数 可以限制 绝对值在3以内

data[np.abs(data)>3]=np.sign(data)*3

#******************************************
df=DataFrame(np.arange(20).reshape(5,4))

#根据轴的长度 产生一组新书序的整数 数组
sampler=np.random.permutation(5)

#take 方法 采纳这些 新的序列

df.take(sampler)

#切掉 前三行 数据

df.take(sampler[0:3])
df.take(np.random.permutation(len(df))[:3])


#**************************************************
#哑变量矩阵

#x，y坐标  有就置1  有点像 真值图

df=DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})

pd.get_dummies(df['key'])

#字段前面 + 字符串
dummies=pd.get_dummies(df['key'],prefix='key')

#拼盘 DataFrame
df_with_dummy=df[['data1']].join(dummies)

#举例 处理ch02 中的 电影目录    统计

mnames=['movie_id','title','genres']


#'::'分割
movies=pd.read_table('movies.dat',sep='::',header=None,names=mnames)
#‘|’ 分割 条目
genre_iter=(set(x.split('|')) for x in movies.genres)
# 取地址 按字母顺序 排列
genres=sorted(set.union(*genre_iter))

#*********************
dummies=DataFrame(np.zeros((len(movies),len(genres))),columns=genres)

# 将每一部 电影的dummies 行 赋 1

# 循环 写法 for 别名 in (a,b): 操作
for i,gen in enumerate(movies.genres):
    dummies.ix[i,gen.split('|')]=1

movies_windic=movies.join(dummies.add_prefix('Genre_'))

movies_windic.ix[0]

#分区间 统计

values=np.random.randn(10)

values

#区间 划分
bins=[0,0.2,0.4,0.6,0.8,1]

pd.get_dummies(pd.cut(values,bins))


#************************************************************
#对字符串的处理

val='a,b,  guido'

#逗号 分割
val.split(',')

#修减 空格

pieces=[x.strip() for x in val.split(',')]

#以指定形式 分割

first,second,third=pieces

first+'::'+second+'::'+third

#更地道  的做法

'::'.join(pieces)

#统计字符串 出现次数

val.count(',')

# 替换

val.replace(',','::')


#************************************************************
#  正则表达式 regex

#re模块  三个类 ：模式匹配   替换  拆分

import re

text="foo bar\t baz \tqux"


# '\s+' 代表了所有 分割空格 不管是 空几个 或是 字符表示
re.split('\s+',text)

#多次 调用的话 可以 先编译  以提高效率
regex=re.compile('\s+')

#findall 找出 所有匹配 正则式 的项
regex.findall(text)

#还有 像 r'C;|x' 前面有个r 代表 启用了 正则式

# 举例  匹配电子邮件格式

text="""Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""

#字母  数字 . _ % + - 符号      {2,4} 二到四 个字符
pattern=r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'

#IGNORECASE 忽略 大小写。
regex=re.compile(pattern,flags=re.IGNORECASE)

regex.findall(text)

#如果使用search 函数 它会搜寻 第一个匹配的位置 并返回位置的地址

m=regex.search(text)

# 打印m位置 的信息

text[m.start():m.end()]

#而regex.match函数 是从字符串开头匹配 不匹配的话返回None

print regex.match(text)

#替换字符 函数 regex.sub

print regex.sub('REDACTED',text)

#若想分段 匹配的字符串 用圆括号 () 包起

pattern=r'([A-Z0-9.%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'


regex=re.compile(pattern,flags=re.IGNORECASE)
kk='wesm@bright.net'
m=regex.match(kk)

kk[m.start():m.end()]

m.groups()

regex.findall(text)

#还可以 给 分离 出 的 每一项 加上 小标题
print regex.sub(r'Username:\1,Domain:\2,Suffix:\3',text)

#还可以 以 字典 的 方式 返回 满足条件 的 项
regex=re.compile(r"""(?P<Username>[[A-Z0-9.%+-]+])@(?P<domain>[A-Z0-9.-]+)\.(?P<suffix>[A-Z]{2,4})""",flags=re.IGNORECASE|re.VERBOSE)

m=regex.match(' ')

#****************************************************
P222

import  numpy as np
from pandas import Series,DataFrame
data={'Dave':'dave@google.com','Steve':'steve@gmail.com','Rob':'rob@gmail.com','Wes':np.nan}

data=Series(data)

#*******************************************************
#实际例子 分析 USDA食品数据库 P224

import json
import pandas as pd

db=json.load(open('ch07/foods-2011-10-03.json'))

db[0].keys()


nutrients=DataFrame(db[0]['nutrients'])

nutrients[:7]

info_keys=['description','group','id','manufacturer']
info=DataFrame(db,columns=info_keys)
info[:5]

#value_counts 查看食物类别

pd.value_counts(info.group)[:10]


# 整合表
#   1          将各种营养成分转化成 DataFrame表
#   2           将该DataFrame 表 放到nutrients 列表中
#   3           最后使用 concat 函数把 对应的东西连接起来


nutrients=[]
for rec in db:
    fnuts=DataFrame(rec['nutrients'])
    fnuts['id']=rec['id']
    nutrients.append(fnuts)

nutrients=pd.concat(nutrients,ignore_index=True)

#去重复

nutrients.duplicated().sum()

nutrients.drop_duplicates()

#重命名 类目
col_mapping={'description':'food',
             'group':'fgroup'}
info=info.rename(columns=col_mapping,copy=False)

col_mapping={'description':'nutrient',
             'group':'nutgroup'}

nutrients=nutrients.rename(columns=col_mapping,copy=False)

#合并info 和nutrient
ndata=pd.merge(nutrients,info,on='id',how='outer')


#分析数据 取中位值


result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)

#锌元素
result['Zinc, Zn'].order().plot(kind='barh')

get_maximum=lambda x:x.xs(x.value.idxmax())
get_minimum=lambda x:x.xs(x.value.idxmin())

max_foods=by_nutrient.apply(get_maximum)[['value','food']]

max_foods.food=max_foods.food.str[:50]

max_foods.ix['Amino Acids']['food']
























































































