# deepspeed-CML

### Single Worker:

- Code snippet:
```
per_device_train_batch_size = 8
```

- During Training
<img width="1013" alt="image" src="https://github.com/dennislee22/deepspeed-LLM-FT-CML/assets/35444414/98dace1e-b517-4303-ab9e-57dba16fb40c">

```
{'loss': 0.8018, 'learning_rate': 0.0001936370577755154, 'epoch': 2.03}
{'loss': 0.6509, 'learning_rate': 0.0001935522185458556, 'epoch': 2.03}
{'loss': 0.7121, 'learning_rate': 0.00019346737931619584, 'epoch': 2.03}
{'train_runtime': 703.2716, 'train_samples_per_second': 33.52, 'train_steps_per_second': 16.76, 'train_loss': 0.8185263499425055, 'epoch': 2.03}
Training Done
Model Saved
```

- Code snippet:
```
per_device_train_batch_size = 1
```

- During Training
<img width="1012" alt="image" src="https://github.com/dennislee22/deepspeed-LLM-FT-CML/assets/35444414/8e999919-b18a-4068-96af-98f0a7c9afc8">

```
{'loss': 0.7819, 'learning_rate': 4.83880546364639e-05, 'epoch': 2.03}
{'loss': 0.763, 'learning_rate': 4.837744973275643e-05, 'epoch': 2.03}
{'loss': 0.6486, 'learning_rate': 4.836684482904896e-05, 'epoch': 2.03}
{'train_runtime': 721.1238, 'train_samples_per_second': 32.691, 'train_steps_per_second': 32.691, 'train_loss': 0.879957252825899, 'epoch': 2.03}
Training Done
Model Saved
```


