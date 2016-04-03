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

mean_ratings=data.pivot_table('rating',rows='title',cols='gender',aggfunc='mean')


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

#******************************************
#爬虫抓取，原地址：http://python.jobbole.com/84349/

#-*-coding:utf8-*-
#
# import re
# import string
# import sys
# import os
# import urllib
# import urllib2
# from bs4 import BeautifulSoup
# import requests
# from lxml import etree
#
# reload(sys)
# sys.setdefaultencoding('utf-8')
# if(len(sys.argv)&gt;=2):
#     user_id = (int)(sys.argv[1])
# else:
#     user_id = (int)(raw_input(u"请输入user_id: "))
#
# cookie = {"Cookie": "#your cookie"}
# url = 'http://weibo.cn/u/%d?filter=1&amp;page=1'%user_id
#
# html = requests.get(url, cookies = cookie).content
# selector = etree.HTML(html)
# pageNum = (int)(selector.xpath('//input[@name="mp"]')[0].attrib['value'])
#
# result = ""
# urllist_set = set()
# word_count = 1
# image_count = 1
#
# print u'爬虫准备就绪...'
#
# for page in range(1,pageNum+1):
#
#   #获取lxml页面
#   url = 'http://weibo.cn/u/%d?filter=1&amp;page=%d'%(user_id,page)
#   lxml = requests.get(url, cookies = cookie).content
#
#   #文字爬取
#   selector = etree.HTML(lxml)
#   content = selector.xpath('//span[@class="ctt"]')
#   for each in content:
#     text = each.xpath('string(.)')
#     if word_count &gt;= 4:
#       text = "%d :"%(word_count-3) +text+"\n\n"
#     else :
#       text = text+"\n\n"
#     result = result + text
#     word_count += 1
#
#   #图片爬取
#   soup = BeautifulSoup(lxml, "lxml")
#   urllist = soup.find_all('a',href=re.compile(r'^http://weibo.cn/mblog/oripic',re.I))
#   first = 0
#   for imgurl in urllist:
#     urllist_set.add(requests.get(imgurl['href'], cookies = cookie).url)
#     image_count +=1
#
# fo = open("/Users/Personals/%s"%user_id, "wb")
# fo.write(result)
# word_path=os.getcwd()+'/%d'%user_id
# print u'文字微博爬取完毕'
#
# link = ""
# fo2 = open("/Users/Personals/%s_imageurls"%user_id, "wb")
# for eachlink in urllist_set:
#   link = link + eachlink +"\n"
# fo2.write(link)
# print u'图片链接爬取完毕'
#
# if not urllist_set:
#   print u'该页面中不存在图片'
# else:
#   #下载图片,保存在当前目录的pythonimg文件夹下
#   image_path=os.getcwd()+'/weibo_image'
#   if os.path.exists(image_path) is False:
#     os.mkdir(image_path)
#   x=1
#   for imgurl in urllist_set:
#     temp= image_path + '/%s.jpg' % x
#     print u'正在下载第%s张图片' % x
#     try:
#       urllib.urlretrieve(urllib2.urlopen(imgurl).geturl(),temp)
#     except:
#       print u"该图片下载失败:%s"%imgurl
#     x+=1
#
# print u'原创微博爬取完毕，共%d条，保存路径%s'%(word_count-4,word_path)
# print u'微博图片爬取完毕，共%d张，保存路径%s'%(image_count-1,image_path)
#
# python web_spider.py 2549228714
#**********************

































