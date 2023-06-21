#有兩個相同交易頻率的策略A和B，策略A的勝率是66.67%，盈虧比是1.5，策略B的勝率是33.33%，盈虧比是4.2
#請問在經過相同時間長度的充分交易（例如交易10000次）後，哪個策略的預期收益更高？

import random
import matplotlib
import matplotlib.pyplot as plt

n_trade = 10000 

a_t_profit = 0
a_t_profit_list = []
b_t_profit = 0
b_t_profit_list = []

x_list = list(range(n_trade))
for i in x_list:

#策略A勝率66.66%
    a_profit = random.choice([True, True, False])
    if a_profit:
        a_t_profit = a_t_profit + 1.5
    else:
        a_t_profit = a_t_profit - 1.0
    a_t_profit_list.append(a_t_profit)
    
 #策略B勝率33.33%
    b_profit = random.choice([True, False, False])
    if b_profit:
        b_t_profit = b_t_profit + 4.2
    else:
        b_t_profit = b_t_profit - 1.0
    b_t_profit_list.append(b_t_profit)

print('%d次交易後，策略A之報酬率%.1f，策略B之報酬率%.1f'%(n_trade,a_t_profit,b_t_profit))
if a_t_profit > b_t_profit:
    print("A strategy win!")
elif  a_t_profit == b_t_profit:
    print("Tied!")
else :
    print("B strategy win!")

plt.figure(figsize=(8,6))
plt.scatter(x_list, a_t_profit_list, label='Strategy A Profit')
plt.scatter(x_list, b_t_profit_list, label='Strategy B Proft')
plt.legend()
plt.show()
