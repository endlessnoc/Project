#有A、B兩支球隊，球隊A只投兩分球，百發百中，命中率100%。球隊B只投三分球，但投三中二，命中率66.67%
#請問A、B這兩個隊伍打一場比賽，誰的贏面更大？
#每個球隊進球之後，都會由對方發球。（球隊B投球會有搶籃板的問題）
#此問題假設兩個隊伍的搶籃板能力相同，不考慮抄截罰球等問題。

import random

# 球隊A得分和出手次數
a_score = 0
a_control_times = 0

# 球隊B得分和出手次數
b_score = 0
b_control_times = 0
# 模擬球隊B的命中率
goal_list = [True, True, False]

# 隨機選擇哪只球隊開球
control_ball = random.choice(['A','B'])
for i in range(100):
    if control_ball == 'A':
        a_control_times = a_control_times + 1
        a_score = a_score + 2
        print('%d. 球隊[A]出手，命中，球隊A和球隊B當前比分為 %d ：%d'%(
              i+1, a_score, b_score)) 
        control_ball = 'B'
    else:
        b_control_times = b_control_times + 1
        goal = random.choice(goal_list)
        if goal:
            b_score = b_score + 3
            print('%d. 球隊[B]出手，命中，球隊A和球隊B當前比分為 %d ：%d'%(
                  i+1, a_score, b_score)) 
            control_ball = 'A'
        else:
            print('%d. 球隊[B]出手，投失，球隊A和球隊B當前比分為 %d ：%d'%(
                  i+1, a_score, b_score)) 
            # 模擬搶籃板
            control_ball = random.choice(['A','B'])    

print('比賽結束！')
print('球隊A累計出手次數：%d，最終得分：%d'%(a_control_times, a_score))
print('球隊B累計出手次數：%d，最終得分：%d'%(b_control_times, b_score))
if a_score> b_score: 
    print("A team Win!")
elif a_score == b_score:
    print("Tied!")
else:
    print("B team Win!")
