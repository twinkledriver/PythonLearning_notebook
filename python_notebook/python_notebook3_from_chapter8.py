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

#曲线的颜色调整
plt.figure()
plt.plot(randn(30).cumsum(),'--',color='g')







