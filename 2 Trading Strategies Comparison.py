#有兩個相同交易頻率的策略A和B，策略A的勝率是66.67%，盈虧比是1.5，策略B的勝率是33.33%，盈虧比是4.2
#請問在經過相同時間長度的充分交易（例如交易10000次）後，哪個策略的預期收益更高？
#There are two strategies, A and B, with the same trading frequency. 
#Strategy A has a win rate of 66.67% and a profit-loss ratio of 1.5.
#Strategy B has a win rate of 33.33% and a profit-loss ratio of 4.2.
#After a sufficient number of trades over the same period (E.g., 10,000 trades), which strategy would yield higher expected returns?

import random
import matplotlib
import matplotlib.pyplot as plt

N_trade = 10000 

A_t_profit = 0
A_t_profit_list = []
B_t_profit = 0
B_t_profit_list = []

x_list = list(range(n_trade))
for i in x_list:

#Strategt A Win Rate = 66.66%
    A_profit = random.choice([True, True, False])
    if A_profit:
        A_t_profit = A_t_profit + 1.5
    else:
        A_t_profit = A_t_profit - 1.0
    A_t_profit_list.append(A_t_profit)
    
 #Strategt B Win Rate = 33.33%
    B_profit = random.choice([True, False, False])
    if B_profit:
        B_t_profit = B_t_profit + 4.2
    else:
        B_t_profit = B_t_profit - 1.0
    B_t_profit_list.append(B_t_profit)

print('After %d Times of Trading, Strategt A Return = %.1f，Strategy B Return = %.1f'%(N_trade,A_t_profit,B_t_profit))
if A_t_profit > B_t_profit:
    print("A strategy win!")
elif  A_t_profit == B_t_profit:
    print("Tied!")
else :
    print("B strategy win!")

plt.figure(figsize=(8,6))
plt.scatter(x_list, A_t_profit_list, label='Strategy A Profit')
plt.scatter(x_list, B_t_profit_list, label='Strategy B Proft')
plt.legend()
plt.show()
