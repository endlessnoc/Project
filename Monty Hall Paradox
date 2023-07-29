import random

NoC=0 #不換門
c=0 #換門

for i in range(10000): #做1000次實驗
    car = random.choice([1,2,3]) #汽車隨機出現在三扇門之中
    player = random.choice([1,2,3]) #挑戰者選擇了一扇門(隨機)
    #以下數行=主持人隨機選擇一門，這扇門不是挑戰者選擇的門，也不是汽車所在的門
    door_list = [1,2,3]
    door_list.remove(player)
    if car in door_list:
        door_list.remove(car)
    host = random.choice(door_list)
    
    if player == car:
        NoC = NoC+1 #挑戰者原本選的門後有汽車，那不換是對的
    else:
        c =c+1 #挑戰者原本選的門後沒有汽車，那換是對的
    
print("不換：",NoC)
print("換：",c)


