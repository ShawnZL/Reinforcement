# Sarsa

Sarsa 算法与 Q-Learning 算法相似，也是利用 Q 表来选择动作，唯一不同的是两者 Q 表的更新策略不同。该算法由于更新一次动作值函数需要用到 5 个量 ( s , a , r , s ′ , a ′ ) ，所以被称为 Sarsa 算法。

跟新Q表的公式如下所示
$$
Q(S_t,A_t)<-Q(S_t,A_t)+α[R_{t+1}+γ*Q(S_{t+1},A_{t+1})-Q(S_t,A_t)]
$$
其中 Q(St+1, At+1)是下一时刻的状态和实际采取的行动对应的 Q 值；而在 Q-Larning 中是下一时刻的状态对应的 Q 值的最大值，但是在实际中可能不采用该最大值对应的动作。Sarsa 算法和 Q-Learning 算法除了在 Q-target 上有所不同，其他的都一样。

Sarsa 算法是on-policy学习算法，它只有一个策略，使用贪婪选择出Q(St,At) 和 Q(St+1′, At+1′) 。但是Q_learning是off-policy算法，Q(St,At) 使用贪婪算法，计算下一步的时候选择最大值。

## Sarsa算法流程



![](https://github.com/ShawnZL/Reinforcement/raw/master/picture/Sarsa_pic1.png)

初始化 Q 表（令其值为 0）

对于每个 episode（回合）：

 1. 初始化状态 s

 2. 在当前状态 s 的所有可能动作中选取一个动作 a （以 ϵ \epsilonϵ 的概率安装 Q 表数值最大的动作行动，以 1 − ϵ 1-\epsilon1−ϵ 的概率随机行动）

 3. 如果当前状态 s 不是终止状态，则重复执行以下步骤：

 （1）执行动作 a 并得到下一个状态 s‘ 和相应的奖励 r

 （2）在当前状态 s’ 的所有可能动作中选取一个动作 a’

 （3）更新 Q 表：Q ( s , a ) ← Q ( s , a ) + α [ r + γ ⋅ Q ( s ′ , a ′ ) − Q ( s , a ) ] 

 （4）更新状态和动作：s=s’， a=a’

```python
# Sarsa
def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_couter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_couter)
        A = choose_action(S, q_table) # 首先根据随机动作选择一个动作
        # A 代表下一个行为，一定要先放置在外边
        while not is_terminated:
            S_, R = get_env_feedback(S, A) # 下一个的动作，奖励
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                A_ = choose_action(S_, q_table)
                q_target = R + GAMMA * q_table.loc[S_, A_]
                # 注意Q_learning 选择的方法是选择最大值
                # 但是Sarsa选择的是还是贪婪算法选择行为
                """
                q_target = R + GAMMA * q_table.iloc[S_,:].max()
                """
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)

            S = S_
            A = A_
            update_env(S, episode, step_couter + 1)
            step_couter += 1
    return q_table

```



# Sarsa-lambda

Q-Learning 和 Sarsa 都是在得到奖励后只更新上一步状态和动作对应的 Q 表值，是单步更新算法，也就是 Sarsa(0)。但是在得到当前奖励值之前所走的每一步（即一个轨迹）都多多少少和最终得到的奖励值有关，所以不应该只更新上一步状态对应的 Q 值。于是就有了多步更新算法——Sarsa(n)。当 n 的值为一个回合（episode）的步数时就变成了回合更新。对于多步更新的 Sarsa 算法我们用 Sarsa(λ) 来统一表示，其中 λ 的取值范围是 [ 0 , 1 ]，其本质是一个衰减值。

![](https://github.com/ShawnZL/Reinforcement/raw/master/picture/Sarsa_pic2.png)

Sarsa(λ) 算法比Sarsa 算法中多了一个矩阵E (eligibility trace)，它用来保存在路径中所经历的每一步，并其值会不断地衰减。该矩阵的所有元素在每个回合的开始会初始化为 0，如果状态 s 和动作 a 对应的 E(s,a) 值被访问过，则会其值加一。并且矩阵 E 中所有元素的值在每步后都会进行衰减，这保证了离获得当前奖励越近的步骤越重要，并且如果前期智能体在原地打转时，经过多次衰减后其 E 值就接近于 0 了，对应的 Q 值几乎没有更新。

值得注意的是，在更新 Q(s,a) 和 E(s,a) 时，是对“整个表”做更新，但是因为矩阵 E 的初始值是 0，只有智能体走过的位置才有值，所以并不是真正的对“整个表”做更新，而是更新获得奖励值之前经过的所有步骤。而那些没有经过的步骤因为对应的 E(s,a) 值为0，所以 Q(s,a) = Q(s,a) + α⋅δ⋅E(s,a) = Q(s,a) ，会保持原值不变。
