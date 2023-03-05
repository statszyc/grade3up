# One

## Str

``` python
<str>.title()
<str>.upper()
<str>.lower()
<str>.rstrip()
<str>.lstrip()
<str>.strip()
r"\n"
"hello"+" "
<str>[0:4:2]
#Can't change the value
```

## Else

``` python
3/2
3//2
3**2
True
False
~(cond)
1!=2
<list>=["a",2,True]
<list>[-1]
<list>.append("a")
<list>.insert(0,"f")
del obj
<list>.pop(0)
<list>.remove("s")
<list>.sort(reverse=True,key=afunc)
sorted(<list>) #copy style
<list>.reverse()
len(<list>)
<list>.index(value)
<list1>+<list2>
type(obj)
obj.copy()
obj.view()
<tuple>=(400,[1,2])
a1,a2,a3=a3,a2,a1
```

# Two

``` python
range(0,10,2) # zuobiyoukai
range(0,10,-1)
```

## Set

```python
<set>={1,2,3}
1 in <set>
<set>=set("jifojewo")
s1-s2
s1|s2
s1&s2
s1^s2
<set>.add(2)
<set>.remove(2)
frozenset('efwf')
```

## dict

``` python
<dict>={"one":1,"two":2}
<dict>["one"]
void={}
<dict>=dict(one=1,two=2)
del <dict>["one"]
<dict>.keys()
<dict>.values()
<dict>=dict(zip(<list1>,<list2>))
<dict1>.update(<dict2>)
```

## loop

``` python
for item in agg:
    if cond1:
        break
else:
    "I did not experience the break!"

for i in range(10):
    pass
print(i)  # i=9
for key,value in <dict>.items():
    pass
while cond:
    if cond1:
        continue
    else:
        break
<listloop>=[func(x) for x in range(somenum)]
<setloop>={func(x) for x in range(somenum)}
<binaryloop>=[func(x,y) for x in ... for y in ... if cond]
```

## if

``` python
if cond:
    pass
elif cond:
    pass
else:
    pass
```

## func

``` python
def func(para):
    """help here"""
    return(any)

def onestar(*atuple):
    for items in atuple:
        print(items)

def twostars(**adict):
    for key,value in adict.items():
        print(key,value)

funs=[func,onestar,twostars]
funs[0]("hh")
lambda x:func(x)
```

## try

``` python
try:
    something
except TypeError:
    something
```

# Three

``` python
class Dog():
    classatttr="my"
    classattr2=[]
    __private2=0
    def __init__(self,para1):
        self.attr=para1
        self.__private=0 #can't visit and 
        #change outside of the class
    def func1(self):
        self.attr=1
    def counts():  # func of the class
        print(Dog.__private2)
myinst=Dog(6)
myinst.para1
myinst.func1()

myinst.classattr # can't change that value here
Dog.classattr="our"

myinst.classattr2.append("red")

myinst._Dog__count #not recommend
class child(father):
    def __init__(self,fa1,fa2,child1,child2):
        super().__init__(fa1,fa2)
        self.para=child1
        self.para=child2
    def samenamewithfather(self):
        just write here
        use this prior than the father one

isinstance(<inst>,<class>)
issubclass(<class1>,<class2>)
```

## namespace

``` python
def scope():
    spam="test"
    def do_nonlocal:
        nonlocal spam
        spam="nonlocal"
    def do_global():
        global spam
        spam="global"
```

# Four

## path

``` python
import os
os.path.join(<strs>,...)
os.path.splitext(<str>)
os.getcwd()
os.chdir(path)
os.listdir(apath)
os.path.exists(path)
os.path.isdir(path)
os.path.isfile(path)
```

## write

``` python
obj=open(path)
obj.close()

with open(path) as obj:
    obj.read()

with open(path) as obj:
    obj.readlines()

with open(path,'w' or 'a') as obj:
    obj.write(content)
```

## save

``` python
import json
with open(path,'w' or 'a') as obj:
    json.dump(content,obj)

with open(path) as obj:
    json.load(obj)

import pickle
# The same, but can dump instance

import shelve
#KVP
shelfile=shelve.open(path)
shelfile[K]=V
shelfile.close()
```

# organize

``` python
import shutil
shutil.copy(path1,path2)
shutil.copytree(path1,path2)
shutil.move(path1,path2)

import os
os.walk(path)
```

# Five(numpy)

## basic

``` python
import numpy as np
np.arrange(50)
<arr>.shape
<arr>.dtype
<arr>.astype(elementtype)
np.random.randn(2,3)
np.array([...])
np.array([[...],[...]])
<arr>.reshape(<tuple>)
<arr>.ndim
np.zeros()
np.empty()
np.ones_like(<arr>)
np.full_like(<arr>,<num>)
np.identity()=np.eye()
*
-
/
**
>
```

## index

``` python
<arr>[4:8]=12
np.array([[[1,2],[1,2]],[[2,3],[2,3]],[the same]])
[0][1][1]
# from outside to inside
<arr>[:,:,0]
<arr>[:,:,:1]
布尔索引和神奇索引返回拷贝，切片返回视图
<arr>[[1,2,3],[2,3,4]]
<arr>[[1,2,3]][:,[2,3,4]]
```

## transform

``` python
<arr>.T
<arr>.transpose(<tuple>)
<arr>.reshape(<tuple>)
np.vstack((<arr1>,<arr2>))
np.hstack((<arr1>,<arr2>))
np.split(<arr>,[nums...])
```

## common method

``` python
np.sqrt()
np.exp()
np.maximum(<arr1>,<arr2>) #逐位取较大，没有方法
np.sin()
np.modf()  #返回remainder,int_part，没有方法
np.mean()
np.std()
np.max()
其他都有方法，且可传入axis
```

## else

``` python
from numpy.linalg import inv
from numpy.linalg import det

np.random.shuffle()
np.random.seed()
np.random.normal(loc,scale,size)
np.random.chisquare()
np.random.binomial()

import random
random.randint(start,end)
```

# Six(pandas)

## Series

``` python
pd.Series([...],index=...)#create series based on the given ind
pd.Series({KVP})
<series>.values
<series>.index
<series>['a']
<series>.[['a','v']] #fancy index
pd.isnull()
pd.notnull()
<series1>+<series2>
<series>.name=...
<series>.index.name=...
```

## DataFrame

``` python
pd.DataFrame({a list after a key},columns=...)
#create df based on the given columns
pd.DataFrame({k1:{kk1:v1,kk2:v2,...},...})
#以k1等为列名，以kk1等为行名建立df
<df>[0][0] # can get a value: 先行后列
<df>["column name"]
<df>.columnname
<df>["columnname"]=<series> #按index赋值
<df>.T
```

Index is unchangable.
`pd.Index`

## operations

``` python
<series>.reindex(newind,method,columns=newcol(选))
    method="ffill" or "bfill"
<series>.drop(values,axis=0)
#return views
<series>[2:4] #索引出第2至3行，对df也一样
<series>['b':'c'] #左闭右闭
#return copies
<series>[['a','b']]
<df>.loc[ind,col] #左闭右闭
<df>.iloc[ind,col] #左闭右开
<df1>.sub(<df2>,fill_value=val,axis)#两边都为nan时仍为
#nan
div, add, rdiv

<df>.apply
<df>.applymap
<series>.map=<series>.apply
<obj>.sort_index(axis,ascending)
<obj>.sort_values(by)
<obj>.rank(method,axis)
    method="first" or "rank" or "max" or null

<obj>.index.is_unique
df.sum(axis,skipna=)
<df>.idxmax(axis)
<series>.unique() #去重
<series>.duplicated() #只有第一次出现返回False
<df>[somecol][<df>[somecol].duplicated().values==0] 
#返回各重复值第一次出现的位置
<series>.value_counts(sort=)
<series>[<series>.isin(<list>)] 
#对series逐元素判断在不在list里面
df.agg(<list> or <dict>)
df.cumsum() #类似求经验分布函数
df.cumcount() #对同一类的计数
```

# Seven.one

``` python
pd.read_csv(path,header=bool,names=<list>,
index_vol=nameofsomelists,,sep='\s+',
skiprows=<list>,na_values=<list>,nrows=<int>,
chunksize=<int>,keep_defualt_na=bool)

<df>.to_csv(na_rep=<str>,index=bool,header=bool,
columns=<list>)

pd.date_range(<str>,periods=<int>)
pd.read_pickle
<df>.to_pickle
pd.HDFStore
<obj>.to_hdf()
pd.read_hdf()
pd.read_excel()
obj.to_excel()
<df>.info()
np.count_nonzero()
obj.fillna(<num> or method)
obj.interpolate()#上下两值的均值进行插补
```

# Seven.two

``` python
<series>.str.lower()
<series>.map(<dict>)=<series>.map(lambda x:<dict>[x.lower()])
<series>.replace()
<df>.rename(index=str.title,columns=str.upper)
<cat>=pd.cut(<list>,bins,right=True,labels=<list>,precision=<int>)
    right=True左开右闭, right=False左闭右开
cat.codes
cat.categories

<cat>=pd.qcut(<list>,<quantiles>)
<boolseries>.any(axis)
<boolseries>.all(axis)
any(<list>)
all(<list>)
sampler=np.random.permutatin(5) #生成一个0~4的随机排列
<df>.take(sampler) #进行行置换

<df>.sample(n=<int>,replace=True) #有放回地抽n个
<series>.unstack()
<df>.stack()
<df>.swaplevel(<str1>,<str2>)
<df>.sort_index(level=<int>) or sum
df.set_index([columnnames],drop=True)
    drop是否把去当index的column仍保留在value中
df.reset_index() #还原set_index之前的，但drop为False时不可用
```

merge, join, groupby自己看笔记

# Eight

## plt

``` python
plt.plot(xdata,ydata,'o',)
fig=plt.figure()
fig.suptitle(<str>)
fig.tight_layout()

ax=fig.add_subplot(221)
ax.set_title(<str>)
ax.set_xlabel(<str>)
ax.set_ylabel(<str>)

ax.plot(...)
ax.hist(data,bins)
ax.scatter(xdata,ydata,s=...,c=...,alpha=<float>)
ax.boxplot([datas],labels=[<strs>])
```

## seaborn

``` python
fig,ax=plt.subplots()

sns.distplot(data,kde=bool,hist=bool,rug=bool)
sns.countplot(x,data)
sns.kdeplot(x,y,data,shade=bool)
sns.barplot(x,y,data)
sns.boxplot(x,y,data,hue) #对x的二级分类变量
sns.violinplot(x,y,data,hue,split=bool)

#回归散点图
sns.regplot(x,y,data)#创建一个axes
sns.lmplot(x,y,data,hue,fit_reg=True,markers=[strs])#创建一个figure
    fit_reg=False时就不加回归线，变成了散点图

#联合图or成对关系
sns.jointplot(x,y,data,kind=<str>)
    kind="sex" or "kde"
sns.pairplot(data,hue)
pg=sns.PairGrid(data)
pd.map_upper(sns.funcobj)
pd.map_lower(sns.funcobj)
pd.map_diag(sns.funcobj)

#分面
sns.lmplot(x,y,data,fit_reg,col=<str>,col_wrap=<int>,
scatter_kws={"s":100})
    col是列分面变量，col_wrap表示排列成几行

fc=sns.FacetGrid(datas,col,hue)
fc.map(sns.funcobj,params)
fc.add_legend()

fc=sns.FacetGrid(datas,col,row,hue)
fc.map(...)

sns.catplot(x,y,data,hue,row,col,kind)

sns.lmplot(x,y,data,hue,fit_reg,row,col,kind)

#style
with sns.axes_style(stylename):
    ...
```

## pandas

``` python
<df[[one or more columns]]>.plot.
hist(alpha,bins,ax)
kde(ax)
scatter(x,y,ax)
hexbin(x,y,gridsize,ax)
box(ax)
```

# Nine

## statsmodels.formula.api

``` python
import statsmodels.formula.api as smf

model=smf.ols(formula,data)
model=smf.logit(formula,data)

results=model.fit()

results.summary()
results.params

```

## sklearn

``` python
from sklearn import linear_model

model=linear_model.LinearRegression(params)
model=linear_model.Lasso(...)

predicted=model.fit(X=,y=) #give data

predicted.coef_
predicted.intercept_
predicted.score(X_test,y_test)
dm=pd.get_dummies(xdata,drop_first=True)
model.fit(dm,ydata)

from patsy import dmatrices
response,predictors=dmatrices(formula,data=df)
model=linear_model.LinearRegression(normalize=bool,fit_intercept=False)
res=model.fit(predictors,response)

displaydf=pd.DataFrame(list(zip(predictors.design_info.column_names,
res.coef_[0])),columns=["variables","coef_lr"])
from sklearn.model_selection import train_test_split
after dmatrices
X_train,X_test,y_train,y_test=train_test_split(predictors,response)
```

## cluster

``` python
from sklearn.cluster import KMeans
kmeans=KMeans(n_cluster=<int>).fit(data)

kmeans.lables_

from sklearn.decomposition import PCA
pca=PCA(n_components=2).fit(data)
pca_trans=pca.transform(data)

#picture

kmeans3=pd.DataFrame(kmeans.labels_,columns=["cluster"])
pca_tr_df=pd.DataFrame(pca_trans,columns=["pca1","pca2"])
kmeanspca=pd.concat([kmeans3,pca_tr_df],axis=1)
sns.lmplot(x="pca1",y="pca2",data=kmeanspca,hue="cluster",fit_reg=False)

kmeanswithlabeldf=pd.concat([kmeanspca,truelabel],axis=1)
sns.lmplot(x="pca1",y="pca2",data=kmeanswithlabeldf,row="cluster",col="truelabel",fit_reg=False)
from scipy.cluster import hierarchy

data_trans=hierarchy.complete(data) or .linkage(data,"complete")
data_trans=hierarchy.single(data)
data_trans=hierarchy.average(data)

hierarchy.dendrogram(data_trans)
```

## KFold

``` python
from sklearn.model_selection import KFold
kf=KFold(n_splits=<int>)
after dmatrices get X and y
for train,test in kf.split(X): #return masks
    X_train,X_test=X[train],X[test]
    y_train,y_test=y[train],y[test]
    lr=<new model>.fit(X_train,y_train)
    testscore.append(lr.score(X_test,y_test))
from sklearn.model_selection import cross_val_score
model=<new model>
after dmatrices get X and y
scores=cross_val_score(model,X,y,cv=<int>)
```
