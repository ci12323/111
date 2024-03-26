#!/usr/bin/env python
# coding: utf-8

# # 让AI开始作诗

# 本AI可以实现以下两种功能：
# 1. 续写诗句
# 2. 创作藏头诗
# 
# 首先，运行下面的代码准备环境

# In[1]:


from sys import path

path.append("work")


# In[2]:


from poet import Poet

AI = Poet()
print("加载完成！")


# 运行下面的代码，**让AI续写诗句**

# In[4]:


print(AI.renewal(input("请输入要续写的诗句：")))


# 运行下面的代码，**让AI创作藏头诗**

# In[3]:


print(AI.acrostic(input("请输入藏头诗的头：")))


# In[ ]:


input("please input any key to exit!")

