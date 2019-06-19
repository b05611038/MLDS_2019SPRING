# Actor-Critic Implementation by pytorch

MLDS HW4-3

## Enviroment

Open AI gym atari game Pong-v0

## Training Actor-Critic Agent

Sample command, please use -h for further information

```
python3 ACAgent_play.py baseline baseline_model A2C -1 
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
A2C(Actor-critic)
A2C_PPO(Actor-critic with Proximal Policy Gradient, with KL Divergence loss)
A2C_PPO2(Actor-critic with Proximal Policy Gradient ver2, clip important sampling weight)
```

## Playing video making

Sample command, please use -h for further information

```
python3 Agent_show.py baseline baseline_model -1
```

### Note

```
Please use the same config as training when testing your model.
```
