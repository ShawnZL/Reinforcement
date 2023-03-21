# Sarsa-lambda

Q-Learning 和 Sarsa 都是在得到奖励后只更新上一步状态和动作对应的 Q 表值，是单步更新算法，也就是 Sarsa(0)。但是在得到当前奖励值之前所走的每一步（即一个轨迹）都多多少少和最终得到的奖励值有关，所以不应该只更新上一步状态对应的 Q 值。于是就有了多步更新算法——Sarsa(n)。当 n 的值为一个回合（episode）的步数时就变成了回合更新。对于多步更新的 Sarsa 算法我们用 Sarsa(λ) 来统一表示，其中 λ 的取值范围是 [ 0 , 1 ]，其本质是一个衰减值。

![](https://github.com/ShawnZL/Reinforcement/raw/master/picture/Sarsa_pic2.png)

Sarsa(λ) 算法比Sarsa 算法中多了一个矩阵E (eligibility trace)，它用来保存在路径中所经历的每一步，并其值会不断地衰减。该矩阵的所有元素在每个回合的开始会初始化为 0，如果状态 s 和动作 a 对应的 E(s,a) 值被访问过，则会其值加一。并且矩阵 E 中所有元素的值在每步后都会进行衰减，这保证了离获得当前奖励越近的步骤越重要，并且如果前期智能体在原地打转时，经过多次衰减后其 E 值就接近于 0 了，对应的 Q 值几乎没有更新。

值得注意的是，在更新 Q(s,a) 和 E(s,a) 时，是对“整个表”做更新，但是因为矩阵 E 的初始值是 0，只有智能体走过的位置才有值，所以并不是真正的对“整个表”做更新，而是更新获得奖励值之前经过的所有步骤。而那些没有经过的步骤因为对应的 E(s,a) 值为0，所以 Q(s,a) = Q(s,a) + α⋅δ⋅E(s,a) = Q(s,a) ，会保持原值不变。