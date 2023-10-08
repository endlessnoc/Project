import random
import matplotlib.pyplot as plt

N_trade = 10000 

A_t_profit = 0
A_t_profit_list = []
B_t_profit = 0
B_t_profit_list = []

x_list = list(range(N_trade))
for i in x_list:

#Strategt A Win Rate = 66.66%, profit-and-loss ratio = 1.5
    A_profit = random.choice([True, True, False])
    if A_profit:
        A_t_profit = A_t_profit + 1.5
    else:
        A_t_profit = A_t_profit - 1.0
    A_t_profit_list.append(A_t_profit)
    
 #Strategt B Win Rate = 33.33%, profit-and-loss ratio = 4.2
    B_profit = random.choice([True, False, False])
    if B_profit:
        B_t_profit = B_t_profit + 4.2
    else:
        B_t_profit = B_t_profit - 1.0
    B_t_profit_list.append(B_t_profit)

#Printing Result
print('After %d Times of Trading, Strategt A Return = %.1f，Strategy B Return = %.1f'%(N_trade,A_t_profit,B_t_profit))
if A_t_profit > B_t_profit:
    print("A strategy win!")
elif  A_t_profit == B_t_profit:
    print("Tied!")
else :
    print("B strategy win!")

#Visulization
plt.figure(figsize=(8,6))
plt.scatter(x_list, A_t_profit_list, label='Strategy A Profit')
plt.scatter(x_list, B_t_profit_list, label='Strategy B Proft')
plt.legend()
plt.show()
