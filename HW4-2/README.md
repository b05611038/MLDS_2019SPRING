# Q Learning Implementation by pytorch

MLDS HW4-2

## Training Q-learning Agent

Sample command, please use -h for further information

```
python3 PGAgent_play.py baseline baseline_model Q_l1_target -1
```

### Implemented Action Space Exploring

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
time_decay
```

### Implmented Q-Learning base Algorithm

```
Q_l1 (Q Learning with smooth L1 loss)
Q_l2 (Q Learning with mean squared loss)
Q_l1_target (Q Learning with smooth L1 loss and target net)
Q_l2_target (Q Learning with mean squared loss and target net)
```

## Playing Video Making

Sample command, please use -h for further information

```
python3 Agent_show.py baseline baseline_model -1
```

### Note

```
Please use the same config as training when testing your model.
```

