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
years=range(1980,2011);
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
total_births.set_ylim(0,2500000)
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















































