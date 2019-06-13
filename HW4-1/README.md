# Policy Gradient Implementation by pytorch

MLDS HW4-1

## Training Policy Gradient Agent

Sample command, please use -h for further information

```
python3 PGAgent_play.py baseline baseline_model PO -1 
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
PO(Policy Gradient)
PPO(Proximal Policy Gradient, with KL Divergence loss)
PPO2(Proximal Policy Gradient ver2, clip important sampling weight)
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
