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










