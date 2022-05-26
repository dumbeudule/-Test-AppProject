#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import statsmodels.api as sm


# In[2]:


#Read the data about Ad_status
ads = pd.read_csv (r'C:\Users\DUONG\Desktop\iKame\[Câu 1][PBI]\Ad_Status.csv')
print (ads)


# Đối với yêu cầu phân tích, ta sẽ quan sát một số biến có thể rút ra từ bộ số liệu

# In[3]:


#Biến 1: tần suất xem quảng cáo của từng người dùng trong game
freq = ads.loc[:, ['user_pseudo_id', 'event_name']].groupby('user_pseudo_id').count()
freq = freq.rename(columns={'event_name': 'user_freq'})
print(freq)


# In[4]:


#Xem tỷ lệ phân phối của biến 1
print('Lượt xem thấp nhất của 1 người dùng: ' + str(min(freq['user_freq'])))
print('Lượt xem cao nhất của 1 người dùng: ' + str(max(freq['user_freq'])))
print('Lượt xem trung bình của 1 người dùng: ' + str(freq['user_freq'].mean()))
hist1 = sns.countplot(x='user_freq',data=freq)
print(hist1)


# In[5]:


#Biến 2: Mức độ cập nhật của phiên bản game - xem phân phối của biến
hist2 = sns.countplot(x='version',data=ads)
print(hist2)


# Ta thấy biến này đang bị chia quá nhỏ dẫn đến chênh lệch đáng kể giữa các phân phối => phân loại lại biến

# In[6]:


#Chia mức độ cập nhật của phiên bản theo thang đo từ 1-6
conditions = [
    (ads['version'].str.startswith('1') == True),
    (ads['version'].str.startswith('2') == True),
    (ads['version'].str.startswith('3') == True),
    (ads['version'].str.startswith('4') == True),
    (ads['version'].str.startswith('5') == True),
    (ads['version'].str.startswith('6') == True)
    ]
values = [1, 2, 3, 4, 5, 6]
ads['ver_upd'] = np.select(conditions, values)
upd = ads.loc[:, ['user_pseudo_id', 'ver_upd']].groupby('user_pseudo_id').first()
print(upd)


# In[7]:


#Xem tỷ lệ phân phối của biến 2
print('Số người dùng xem quảng cáo trong phiên bản cũ nhất (phiên bản 1): ' + str(upd[upd.ver_upd == 1]['ver_upd'].count()))
print('Số người dùng xem quảng cáo trong phiên bản mới nhất (phiên bản 6): ' + str(upd[upd.ver_upd == 6]['ver_upd'].count()))
hist02 = sns.countplot(x='ver_upd',data=upd)
print(hist02)


# In[8]:


#Theo dõi tương quan giữa biến 1 và biến 2
merged1 = pd.merge(left=upd, right=freq, left_on='user_pseudo_id', right_on='user_pseudo_id')
chart1 = merged1.plot.scatter(x ='ver_upd', y ='user_freq', c ='#FFA500')
print(merged1)
print(chart1)


# In[9]:


#Biến 3: Quy mô của thị trường trong ngành


# --Dữ kiện được lấy tại link: https://rlist.io/dataset/budapesto/17736409/newzoo-game-

# Biến này sẽ được đánh giá bằng thang đo từ 1 đến 6, dựa trên xếp hạng quy mô thị trường game theo quốc gia, với các quốc gia có doanh thu trên 10 tỷ đô được đánh giá 6 điểm, các quốc gia có doanh thu trên 1 tỷ đô được đánh giá 5 điểm, các quốc gia có doanh thu trên 400 triệu đô được đánh giá 4 điểm, các quốc gia có doanh thu trên 100 triệu đô được đánh giá 3 điểm, các quốc gia còn lại trong top 100 được đánh giá 2 điểm và các quốc gia ngoài top 100 được đánh giá 1 điểm

# In[10]:


#Read the data about marker ranking
rank = pd.read_excel (r'Desktop\ranking.xlsx')
print(rank)


# In[11]:


#Xem phân phối của biến 3
hist3 = sns.countplot(x='market_size',data=rank)
print(hist3)


# In[12]:


#Theo dõi tương quan giữa biến 1 và biến 3
merged2 = pd.merge(left=ads, right=rank, left_on='country', right_on='country')
ctr = merged2.loc[:, ['user_pseudo_id', 'country', 'market_size']].groupby('user_pseudo_id').first()
ads_per_ctr = pd.merge(left=freq, right=ctr, left_on='user_pseudo_id', right_on='user_pseudo_id')
chart2 = ads_per_ctr.plot.scatter(x ='market_size', y ='user_freq', c ='#FFA500')
print(ads_per_ctr)
print(chart2)


# In[13]:


#Tổng hợp kết quả của các biến 1, biến 2 và biến 3
merged3 = pd.merge(left=merged1, right=ctr, left_on='user_pseudo_id', right_on='user_pseudo_id')
mod = merged3.loc[:, ['ver_upd', 'market_size','user_freq']].groupby('user_pseudo_id').first()
print(mod)


# In[14]:


#Biến 4: Tổng lượt xem quảng cáo theo ngày
daily = ads.loc[:, ['event_date', 'event_name']].groupby('event_date', as_index=False).count()
daily = daily.rename(columns={'event_name': 'total_ads'})
chart3 = daily.plot.line()
print(daily)
print(chart3)


# In[15]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from numpy import log


# In[16]:


daily.set_index('event_date',inplace=True)
result=adfuller(daily.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print(f'Inference: The time series is {"non-" if result[1] >= 0.05 else ""}stationary')


# In[17]:


sm.graphics.tsa.plot_acf(daily['total_ads'].squeeze(), lags =6)
sm.graphics.tsa.plot_pacf(daily['total_ads'].squeeze(), lags =6)
plt.show()


# In[18]:


warnings.filterwarnings("ignore")
mod = sm.tsa.statespace.SARIMAX(daily,
                                order=(1, 0, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[19]:


pred = results.get_prediction(start=14, end=21, dynamic=False)
pred_ci = pred.conf_int()


# In[20]:


ax = daily['total_ads'].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='statistic forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Total ads')
plt.legend()

plt.show()


# In[21]:


daily_forecasted = pred.predicted_mean
print(daily_forecasted)


# In[ ]:




