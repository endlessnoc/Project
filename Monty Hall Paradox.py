#There are 3 doors, behind which are 2 goats and 1 car. You have to select a door first (let's call it door A).
#The goal is to find the car behind it. Monty Hall, the host of the game, inspects the other doors (B & C) and opens one with a goat behind it. (If both doors conceal goats behind, he randomly selects one.)
#Here's the scenario: 
#1. Stick with your initial choice of door A  
#2. Switch to the unopened door? 
# Which choice is better? 


import random

c=0 #Switch
NoC=0 #Not Switch

for i in range(10000): # Number of Simulation
    car = random.choice([1,2,3]) # 1 Car in 3 doors 
    player = random.choice([1,2,3]) # Player randomly select 1 door
    #The host randomly selects a door, which is neither the door chosen by the player nor the one with a car behind.
    door_list=[1,2,3]
    door_list.remove(player)
    if car in door_list:
        door_list.remove(car)
    host = random.choice(door_list)
    
    if player == car:
        NoC = NoC+1 #TPlayer's initial choice is right (There's a car behind), so not switching is better.
    else:
        c =c+1 #Player's initial choice is wrong (There's not a car behind), so switching is better.
    
print("Not Switch Win：",NoC)
print("Switch Win：",c)
print("Win Rate of Switch:", str(100*(c/(c+NoC))),"%")


#Surprisingly, the % are not evenly split at 50-50. If you choose to switch doors, you'll win 2/3 of the time!
