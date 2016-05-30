#《利用python 进行数据分析》  第八章 开始   绘图 和可视化

plot(np.arange(10))

import matplotlib.pyplot as plt

空figure不能 建图

fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)

from numpy.random import randn

plt.plot(randn(50).cumsum(),'k--')
#_ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
_=ax1.hist(randn(100),bins=20,color='k',alpha=0.3)
ax2.scatter(np.arange(30),np.arange(30)+3*randn(30))



#画随机条形图   调整图与图之间的margin
from numpy.random import randn
fig,axes=plt.subplots(2,2,sharex=True,sharey=True)
for i in range(2):
	for j in range(2):
		axes[i,j].hist(randn(500),bins=50,color='k',alpha=0.5)
plt.subplots_adjust(wspace=0,hspace=0)

#曲线的颜色调整 marker下 标记 出该点
plt.figure()
plt.plot(randn(30).cumsum(),'--',color='g')
plt.plot(randn(30).cumsum(),'--',color='#CECB6F',marker='o')
#****************************************************
#点与点 之间默认 是 线性插值的 可以更改
data=randn(30).cumsum()
plt.plot(data,'k--',label='Default')  #默认情况

plt.plot(data,'r--',drawstyle='steps-post',label='steps-post')

plt.legend(loc='best')

#plt.xlim()  #返回当前 x坐标取值范围
plt.xlim([0,35])  #设置   x坐标取值范围



#设置刻度位置,刻度说明
fig=plt.figure();ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum())
ticks=ax.set_xticks([0,250,500,750,1000])
labels=ax.set_xticklabels(['one','two','three','four','five'],rotation=30,fontsize='small') #更改刻度为 想要的字符

ax.set_title('matplotlib plot')  #大标题
ax.set_xlabel('Stages') #横坐标 标题

fig=plt.figure();ax=fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(),label='one')
ax.plot(randn(1000).cumsum(),label='two')          #三条线
ax.plot(randn(1000).cumsum(),label='three')

#不想要 图例的画可以 改 label  
ax.plot(randn(1000).cumsum(),label='_nolegend_')

#自动在合适的位置 添加图例
ax.legend(loc='best')

#给图像添加注解
ax.text(100,10,'Hello World',family='monospace')  # x,y 表示 坐标,family:字体,fontsize 字体大小


#********************************************************
#分析一个 例子  P241


import pandas as pd
from datetime import datetime
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

data=pd.read_csv('ch08/spx.csv',index_col=0,parse_dates=True)
spx=data['SPX']   #读取名叫SPX 的列，放到spx中去

spx.plot(ax=ax,style='k-')  #绘图


#提取三个关键点，后面会着重 标记出
crisis_data=[
(datetime(2007,10,11),'Peak of bull market'),		#date 和  label
(datetime(2008,3,12),'Bear Stearns Fails'),
(datetime(2008,9,15),'Lehman Bankruptcy')
]

#annotate 注解
for date,label in crisis_data:
	ax.annotate(label,xy=(date,spx.asof(date)+50),
	xytext=(date,spx.asof(date)+200),
	arrowprops=dict(facecolor='black'),
	horizontalalignment='left',verticalalignment='top'
	)

ax.set_xlim(['1/1/2007','1/1/2011'])
ax.set_ylim([600,1800])

ax.set_title('Important dates in 2008-2009 finacial crisis')

#***********************************************************
#绘制一些 常规的图形

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

rect=plt.Rectangle((0.2,0.75),0.4,0.15,color='r',alpha=0.7)   #左下角坐标 长 宽  颜色 透明度
circ=plt.Circle((0.7,0.2),0.15,color='b',alpha=0.4)
pgon=plt.Polygon([[0.15,0.15],[0.35,0.4],[0.2,0.6]],color='y',alpha=0.6)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

plt.savefig('figpath.svg')
plt.savefig('figpath.pdf')  #以不同的格式保存 图像

#也可以灵活的 按照dpi 修建空白

#分辨率 400  修剪  空白留白
plt.savefig('figpath.png',dpi=400,bbox_inches='tight')

#可以 把图像写入 别的 文件（不太懂这个功能，似乎是直接从内存读 而不是从磁盘）
from io import StringIO
buffer=StringIO()
plt.savefig(buffer)
plot_data=buffer.getvalue()


#***********************************************************
P244

#还介绍了关于 画图 默认 的设置 如分辨率 x y 初始设置


#高级绘图工具pandas 的使用（之前用的是matplotlib 画一张表 太麻烦）
from pandas import Series,DataFrame

#0到100 10步长
s=Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))


#P247 有很多适用的 画图 常见用法

#忽略Series中index作为横坐标
#s.plot(use_index=False)


#柱状图 barh水平柱状图
#s.plot(kind='bar')

#折线图
#s.plot()

#曲线  x轴取值
s.plot(kind='kde',xlim=[0,5],xticks=[0,1,4])

#*****************************************************************
#randn 正态分布 取值
df=DataFrame(np.random.randn(10,4).cumsum(0),columns=['A','B','C','D'],index=np.arange(0,100,10))
df.plot()


#ax是指 图像 在子图的位置
fig,axes=plt.subplots(2,1)
data=Series(np.random.rand(16),index=list('abcdefghijklmnop'))
data.plot(kind='bar',ax=axes[0],color='k',alpha=0.7,rot=0)
data.plot(kind='barh',ax=axes[1],color='k',alpha=0.7,rot=0)

#*****************************************************************
#用DateFrame来画图

import pandas as pd
from pandas import Series,DataFrame
df=DataFrame(np.random.randn(6,4),index=['one','two','three','four','five','six'],columns=pd.Index(['A','B','C','D'],name='Genus'))
df

#df.plot(kind='bar')
df.plot(kind='bar',stacked=True,alpha=0.5)

tips=pd.read_csv('tips.csv')
party_counts=pd.crosstab(tips.day,tips.size)
party_counts
party_pcts=party_counts.div(party_counts.sum(1).astype(float),axis=0)

party_pcts.plot(kind='bar',stacled=True)

#上面的例子，无法实现。可能是因为 crosstab 已经不再是 笛卡尔积 的输出

#双峰例子(标准正态分布)
comp1=np.random.normal(0,1,size=200)
comp2=np.random.normal(10,2,size=200)

values=Series(np.concatenate([comp1,comp2]))

values.hist(bins=100,alpha=0.3,color='k',normed=True)

values.plot(kind='kde',style='k--')


#散点图

import pandas as pd
from pandas import Series,DataFrame


macro=pd.read_csv('ch08/macrodata.csv')
data=macro[['cpi','m1','tbilrate','unemp']]  #提取数据 中 的 选定项
trans_data=np.log(data).diff().dropna() # 划归对数坐标，去NA
trans_data[-5:]

#绘制散点图 用scatter 函数

#plt.scatter(trans_data['m1'],trans_data['unemp'])
#plt.title('Changes in log %s vs. log %s'%('m1','unemp'))

#也可用pandas 中的DataFrame 的scatter_matrix函数 更加方便
pd.scatter_matrix(trans_data,diagonal='kde',color='k',alpha=0.3)  #对角线以kde 密度 处理


#**************************************************************************************

#一个应用 ：关于图形化 海地地震的数据



#***************************************准备工作*******************************
import pandas as pd
from pandas import Series,DataFrame
data=pd.read_csv('ch08/Haiti.csv')

#data
#data[['INCIDENT DATE','LATITUDE','LONGITUDE']][:10]
#data['CATEGORY'][:6]

#清楚错误 或 缺失 的数据
data=data[(data.LATITUDE>18)&(data.LATITUDE<20)&(data.LONGITUDE>-75)&(data.LONGITUDE<-70)&data.CATEGORY.notnull()]




#定义三个函数 获取所有分类列表   将分类信息拆分成 编码和英语名称
def to_cat_list(catstr):
	stripped=(x.strip() for x in catstr.split(','))  #strip() 函数 去 字符串前后空格，但保留字符串间的空格  split 以‘，’ 分隔
	return [x for x in stripped if x]

def get_all_categories(cat_series):
	cat_sets=(set(to_cat_list(x)) for x in cat_series)
	return sorted(set.union(*cat_sets))

def get_english(cat):
	code,names=cat.split('.')
	if '|' in names:
		names=names.split('|')[1]  # 如果有| 取 | 后面([1])的 作为names
	return code,names.strip()

#get_english('2.Urgences logistiques | Vital Lines  ')

all_cats=get_all_categories(data.CATEGORY)

#all_cats

english_mapping=dict(get_english(x) for x in all_cats)  # 编码 和 名称 的映射 字典

#english_mapping['2a']

#english_mapping['6c']

def get_code(seq):
	return [x.split('.')[0] for x in seq if x]

all_codes=get_code(all_cats)  #all_codes 前面的 a b c 代号
code_index=pd.Index(np.unique(all_codes))			#去重复 
code_index

dummy_frame=DataFrame(np.zeros((len(data),len(code_index))),index=data.index,columns=code_index) # 按行列初始化一个全0 的Dataframe

#dummy_frame.ix[:,:6]

#匹配合适的项    对应置1
for row,cat in zip(data.index,data.CATEGORY):
	codes=get_code(to_cat_list(cat))
	dummy_frame.ix[row,codes]=1

data=data.join(dummy_frame.add_prefix('category_'))


#*****************************************绘图开始*******************************
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def basic_haiti_map(ax=None,lllat=17.25,urlat=20.25,lllon=-75,urlon=-71):
	m=Basemap(ax=ax,projection='stere',lon_0=(urlon+lllon)/2,lat_0=(urlat+lllat)/2,llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,resolution='f')
	m.drawcoastlines()
	m.drawstates()
	m.drawcountries()
	return m

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(12,10))
fig.subplots_adjust(hspace=0.05,wspace=0.05)

to_plot=['2a','1','3c','7a']

lllat=17.25;urlat=20.25;lllon=-75;urlon=-71

for code,ax in zip(to_plot,axes.flat):
	m=basic_haiti_map(ax,lllat=lllat,urlat=urlat,lllon=lllon,urlon=urlon)
	cat_data=data[data['category_%s'%code]==1]
	x,y=m(cat_data.LONGITUDE.values,cat_data.LATITUDE.values)
	m.plot(x,y,'k.',alpha=0.5)
	ax.set_title('%s:%s'%(code,english_mapping[code]))

shapefile_path = 'ch08/PortAuPrince_Roads/PortAuPrince_Roads'
m.readshapefile(shapefile_path, 'roads')

#*****************************************************************
第九章开始 数据聚合与分组运算 


from pandas import DataFrame,Series

df=DataFrame({'key1':['a','a','b','b','a'],
						  'key2':['one','two','one','two','one'],
						  'data1':np.random.randn(5),
						  'data2':np.random.randn(5)})

#按照key1 分组 并计算分组后data1的平均值
grouped=df['data1'].groupby(df['key1'])
grouped.mean()

means=df['data1'].groupby([df['key1'],df['key2']]).mean()

states=np.array(['Ohio','California','California','Ohio','Ohio'])

years=np.array([2005,2005,2006,2005,2006])

df['data1'].groupby([states,years]).mean()



#*****************************************************************

people=DataFrame(np.random.randn(5,5),
									columns=['a','b','c','d','e'],
									index=['Joe','Steve','Wes','Jim','Travis'])
									
people.ix[2:3,['b','c']]=np.nan

mapping={'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}

by_column=people.groupby(mapping,axis=1)

by_column.sum()


#调用自己的创建的函数peak_to_peak 用agg 调用

def peak_to_peak(arr):
	return arr.max()-arr.min()

grouped.agg(peak_to_peak)

#填充缺失值

s=Series(np.random.randn(6))
s[::2]=np.nan

s.fillna(s.mean())

第九章 内容枯燥 需要分组的时候 来查书


接下来联系一个分析联邦选举的例子
P291

#加载一个很大的文件
import pandas as pd
fec=pd.read_csv('ch09/P00000001-ALL.csv')

fec.ix[123456]

unique_cands=fec.cand_nm.unique()

#联系 候选人和 党派
parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

fec.cand_nm[123456:123461]
fec.cand_nm[123456:123461].map(parties)
fec['party']=fec.cand_nm.map()


