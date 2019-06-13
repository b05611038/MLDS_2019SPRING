# Actor Critic Implementation by pytorch

MLDS HW4-3

## Training Actor-critic Agent

Sample command, please use -h for further information

```
python3 ACAgent_play.py baseline baseline_model A2C_l1 -1
```

### Implemented Action space exploring

```
random_action
```


### Implemented Observation Preprocess

```
slice_scoreboard
gray_scale
minus_observation
```

### Implemented Reward Preprocess

```
reward_normalize
decay_by_time
```

### Implmented Policy Gragient base Algorithm

```
A2C_l1(Actor-critic algorithm which critic loss use smooth l1 loss)
A2C_l2(Actor-critic algorithm which critic loss use mean square error loss)
A2C_PPO_l1(Actor-critic algorithm which critic loss use smooth l1 loss and
       combine PPO(Proximal Policy Gradient, with KL Divergence loss) algorithm)
A2C_PPO_l2(Actor-critic algorithm which critic loss use mean square error loss and
       combine PPO(Proximal Policy Gradient, with KL Divergence loss) algorithm)
A2C_PPO2_l1(Actor-critic algorithm which critic loss use smooth l1 loss and
       combine PPO(Proximal Policy Gradient ver2, clip important sampling weight) algorithm)
A2C_PPO2_l2(Actor-critic algorithm which critic loss use mean square error loss and
       combine PPO(Proximal Policy Gradient ver2, clip important sampling weight) algorithm)
```
