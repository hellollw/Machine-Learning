from numpy import *

dic = {'h':1,'b':2,'c':3}
tu = [(1,2),3]
print(type(tu))
b = dic.items() #转换为元组
print(b)
print(type(b))
for i in b:
    print(i)