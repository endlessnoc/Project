#有A、B兩支球隊，球隊A只投兩分球，百發百中，命中率100%。球隊B只投三分球，但投三中二，命中率66.67%
#請問A、B這兩個隊伍打一場比賽，誰的贏面更大？
#每個球隊進球之後，都會由對方發球。（球隊B投球會有搶籃板的問題）
#此問題假設兩個隊伍的搶籃板能力相同，不考慮抄截罰球等問題。
#There are two teams, A and B. 
#Team A only shoots two-point shots and has a perfect shooting record with a 100% FG. 
#Team B only takes three-point shots, giving them a shooting accuracy of 66.67% (2/3).
#After each successful score shot, the ball will turn to the other team. (Only Team B's shooting will results in rebounds.)
#Assumes that 2 teams have equal rebounding abilities and does'nt consider other factors such as stl or FT.


import random
# Team A
a_score = 0
a_control_times = 0

# Team B
b_score = 0
b_control_times = 0
goal_list = [True, True, False]

control_ball = random.choice(['A','B'])
for i in range(100):
    if control_ball == 'A':
        a_control_times = a_control_times + 1
        a_score = a_score + 2
        print('%d. Team[A] Score.\n Scoreboard: \n Team A: TeamB =  %d ：%d'%(
              i+1, a_score, b_score)) 
        control_ball = 'B'
    else:
        b_control_times = b_control_times + 1
        goal = random.choice(goal_list)
        if goal:
            b_score = b_score + 3
            print('%d. Team[B] Score.\n Scoreboard: \n Team A: TeamB = %d ：%d'%(
                  i+1, a_score, b_score)) 
            control_ball = 'A'
        else:
            print('%d. Team[B] Miss.\n Scoreboard: \n Team A: TeamB = %d ：%d'%(
                  i+1, a_score, b_score)) 
            # Rebounds
            control_ball = random.choice(['A', 'B'])    

print('Game End！')
print('Team[A] Number of Shots：%d，Score：%d'%(a_control_times, a_score))
print('Team[B] Number of Shots：%d，Score：%d'%(b_control_times, b_score))
if a_score> b_score: 
    print("A team Win!")
elif a_score == b_score:
    print("Tied!")
else:
    print("B team Win!")
