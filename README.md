# deepspeed Training in <img width="36" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/38692491-f56b-4fc0-a40e-8412b35327b4"> CML 

<img width="407" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/df7fb23c-8828-479c-9a5a-a49fa1969be0">

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Benchmark Score & Summary](#toc_1)<br>
[3. Preparation](#toc_2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Build Custom Docker Image](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. Dataset & Model](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3. CML Session](#toc_5)<br>
[4. Single node with 1 GPU](#toc_6)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Training Result](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Inference](#toc_8)<br>
[5. deepspeed 3 nodes with 1 GPU each (ZeRO 1)](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Training Result](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Inference](#toc_11)<br>
[6. deepspeed 2 nodes with 1 GPU each (ZeRO 1)](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Training Result](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Inference](#toc_14)<br>
[7. deepspeed 3 nodes with 1 GPU each (ZeRO 3)](#toc_15)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Training Result](#toc_16)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.2. Inference](#toc_17)<br>

### <a name="toc_0"></a>1. Objective

1. When fine-tuning/training a LLM, insufficient VRAM is a major constraint. Major components that will be loaded into the VRAM during training process are:

```
VRAM (training/fine-tuning) = Model Parameters + Optimiser + Gradient + Activation 
```

2. For instance, training a model of 1 billon parameters with FP32 would require approximately ~22GB of VRAM.

VRAM (training/fine-tuning) =<br>
<sup>(4bytes * param) + ((4 bytes/param + 4 bytes/param momentum + 4 bytes/param variance) * param) + (4bytes * param) + </sup><img width="300" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4c647806-3634-437b-aba4-d7581437aa59">
 
3. Thus, training a 1B or 7B model with substantial amount of dataset might be able to fit into a single GPU device with 40GB of memory and the latter might need to involve quantization technique when the training takes place. So, the question is how to train a bigger model with billions of parameters given the limited VRAM size. Techniques include Pipeline Parallelism (PP), Data Parallelism (DP) and Tensor Parallelism (TP). This article focuses on **[ZeRO](https://github.com/microsoft/DeepSpeed)** Redundancy Optimizer (ZeRO) technique that shard the 3 model states (optimizer states, gradients, and parameters) across data-parallel processes instead of replicating them (as practised in DP). ZeRO allows fitting huge LLM into the GPUs with restricted memory.

6. The provided iPython codes in this repository serve as a comprehensive illustration of the complete lifecycle for fine-tuning a particular Transformers-based model using specific datasets. This includes merging LLM with the trained adapters, quantization, and, ultimately, conducting inferences with the correct prompt. The outcomes of these experiments are detailed in the following section. The target use case of the experiments is making use the Text-to-SQL dataset to train the model, enabling the translation of plain English into SQL query statements.<br>
&nbsp;a. [ft-trl-train.ipynb](ft-trl-train.ipynb): Run the code cell-by-cell interactively to fine-tune the base model with local dataset using TRL (Transformer Reinforcement Learning) mechanism. Merge the trained adapters with the base model. Subsequently, perform model inference to validate the results.<br>
&nbsp;b. [quantize_model.ipynb](ft-trl-train.ipynb): Quantize the model (post-training) in 8, or even 2 bits using `auto-gptq` library.<br>
&nbsp;c. [infer_Qmodel.ipynb](ft-trl-train.ipynb): Run inference on the quantized model to validate the results.<br>
&nbsp;d. [gradio_infer.ipynb](gradio_infer.ipynb): You may use this custom Gradio interface to compare the inference results between the base and fine-tuned model.<br>

8. Experiments were carried out using `t5-small` and `t5-large` models with 60 million and 770 million parameters respectively in CML v1.5.2 running on Openshift platform. Results are detailed in the following section. 
 
#### <a name="toc_1"></a>2. Summary & Benchmark Score

- Graph below depicts the GPU memory utilization during a specific stage. This graph is computed based on the results obtained from the experiments as detailed in the tables below.
 ZeRO-3 Offload can exploit both GPU and CPU memory,

- Tables below summarize the benchmark result when running the experiments using 1 unit of Nvidia A100-PCIE-40GB GPU on CML with Openshift (bare-metal):<br>

&nbsp;&nbsp;a. Time taken to fine-tune different LLM with 10% of Text-to-SQL dataset (File size=20.7 MB):<br>

| Model     | Fine-Tune Technique | Fine-Tune Duration | Inference Result     |
| :---      |     :---:           |   ---:             | :---                 |
| bloom-1b1  | w/o deepspeed    | ~12 mins           | Good                |
| bloom-7b1  | No Quantization    | OOM                | N/A                  |
| bloom-7b1  | 4-bit BitsAndBytes  | ~83 mins          | Good                 |
| falcon-7b  | No Quantization    | OOM                | N/A                  |
| falcon-7b  | 8-bit BitsAndBytes  | ~65 mins          | Good                 |
| codegen2-1B  | No Quantization    | ~12 mins         | Bad                  |

OOM = Out-Of-Memory

### <a name="toc_2"></a>3. Preparation

#### <a name="toc_3"></a>3.1 Build Custom Docker Image

- Build a Docker image locally (based on the CML image with Jupyter notebook) and push it to the external docker registry, represented by Nexus repository in this example. For reference, CUDA packages can be referenced from this [Nvidia link (ubuntu2004 image)](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/).

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-jupyter
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.1
```

- Build another Docker image locally (based on the CML image with Workbench notebook) and push it to the external docker registry.

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-wb
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
```

- Register the new image in CML.

<img width="1377" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/38c82e3c-2ee4-4e00-9fb1-7a2f2c582779">

- Verify that the image has been registered succesfully.

<img width="694" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/bdc45baa-54a2-4e39-afa1-7e4ff8988192">

#### <a name="toc_4"></a>3.2 Create CML Session

- Create a new CML project with Python 3.10 and GPU variant.

- Add the newly registered image in the CML project.

<img width="1422" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/a88ca709-a10b-43f1-bd30-b9f6786bafbc">

- Add the following environment variables in the CML project.

<img width="1185" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/299b4736-b9fc-4f91-9f8f-09e52bd25f5d">

- Create a new CML session in the project.
  
<img width="1414" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/0ab49111-1b91-4491-9e81-605822a7f84d">

- Open the Terminal window in the CML session and run the following commands to replace the CUDA path with the installed version in the custom docker image.
  
```
$ rm /usr/local/cuda
$ ln -s /usr/local/cuda-12.2 /usr/local/cuda
$ ls -l /usr/local/cuda
lrwxrwxrwx. 1 cdsw cdsw 20 Jan  4 05:38 /usr/local/cuda -> /usr/local/cuda-12.2
```

#### <a name="toc_5"></a>3.3 Create Tensorboard dashboard in CML Application

- Create Tensorboard in the CML application
<img width="476" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/f7a42bef-9c1e-4910-a68b-b9b9961ba831">

- Upon successful creation, browse the Tensorboard website.
<img width="571" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/68b4c50e-b536-458e-ad00-7b67716097af">


#### <a name="toc_6"></a>3.4 Dataset & Model



### <a name="toc_7"></a>4. Single node with 1 GPU (t5-small)

- Batch size 32 is configured for training t5-small model (60 million parameters).
```
!python textsql_train.py \
--model_id 't5-small' \
--outputdir small-trainoutput-no_ds \
--epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_checkpointing False
```

#### <a name="toc_7"></a>4.1 Training Result (t5-small)

<img width="1003" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/17fba932-61a7-4653-9159-bf2f73ace7b4">

- Time taken by single node to complete the training:
```
{'train_runtime': 742.5369, 'train_samples_per_second': 227.686, 'train_steps_per_second': 7.119, 'train_loss': 0.16859441952772136, 'epoch': 3.0}
```

- Tensorboard Profiler (Training + Validation Loss combined):
<img width="1099" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/65ed4421-0ca3-456b-a62a-b4f5806be69b">

#### <a name="toc_8"></a>4.1 Training Result (t5-large)

<img width="1001" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/c7723691-eee7-4b4f-a245-bf151c87a148">

#### <a name="toc_9"></a>4.2 Inference

```
Test Instruction: If you are a pilot officer in the commonwealth then what will you called as in the US air force?
Model Prediction: SELECT US air force FROM table WHERE Pilot Officer = commonwealth
Expected Answer: SELECT US Air Force equivalent FROM table WHERE Commonwealth equivalent = Pilot Officer
=================================

Test Instruction: what is the total number of total w–l where doubles w–l is 11–11
Model Prediction: SELECT COUNT Total W–L FROM table WHERE Doubles W–L = 11–11
Expected Answer: SELECT COUNT Total W–L FROM table WHERE Doubles W–L = 11–11
=================================

Inference took 1.03 seconds
```

### <a name="toc_9"></a>4. deepspeed 3 nodes with ZERO-1 (t5-small)

```
Launch CML worker pod 0...
Launch CML worker pod 1...
Launch CML worker pod 2...
Content of hostfile:
10.254.21.79 slots=1
10.254.18.217 slots=1
10.254.19.152 slots=1
```

- Batch size 32 is configured for training t5-small model (60 million parameters).
```
!export PDSH_SSH_ARGS_APPEND='';deepspeed --hostfile /home/cdsw/hostfile.txt \
--launcher pdsh \
--num_nodes 3 \
--num_gpus 1 \
--master_addr $CDSW_IP_ADDRESS \
--ssh_port 2222 textsql_train.py \
--model_id 't5-small' \
--outputdir ds-zero1-t5small \
--epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--deepspeed dsconfig/zero1profiler.json
```

- DeepSpeed Flops Profiler:
```
-------------------------- DeepSpeed Flops Profiler --------------------------
Profile Summary at step 2:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                                                             3       
data parallel size:                                                     3       
model parallel size:                                                    1       
batch size per GPU:                                                     32      
params per GPU:                                                         60.51 M 
params of model = params per GPU * mp_size:                             60.51 M 
fwd MACs per GPU:                                                       191.26 GMACs
fwd flops per GPU:                                                      382.64 G
fwd flops of model = fwd flops per GPU * mp_size:                       382.64 G
fwd latency:                                                            60 ms   
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:                    6.38 TFLOPS
bwd latency:                                                            287.7 ms
bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency:                2.66 TFLOPS
fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency):      3.3 TFLOPS
step latency:                                                           139.91 ms
iter latency:                                                           487.61 ms
FLOPS per GPU = 3 * fwd flops per GPU / iter latency:                   2.35 TFLOPS
samples/second:                                                         196.88  

----------------------------- Aggregated Profile per GPU -----------------------------
Top 1 modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'T5ForConditionalGeneration': '60.51 M'}
    MACs        - {'T5ForConditionalGeneration': '191.26 GMACs'}
    fwd latency - {'T5ForConditionalGeneration': '59.82 ms'}
depth 1:
    params      - {'T5Stack': '76.96 M'}
    MACs        - {'T5Stack': '140.73 GMACs'}
    fwd latency - {'T5Stack': '52.58 ms'}
depth 2:
    params      - {'ModuleList': '44.06 M'}
    MACs        - {'ModuleList': '140.73 GMACs'}
    fwd latency - {'ModuleList': '49.96 ms'}
depth 3:
    params      - {'T5Block': '44.06 M'}
    MACs        - {'T5Block': '140.73 GMACs'}
    fwd latency - {'T5Block': '49.96 ms'}
depth 4:
    params      - {'ModuleList': '44.06 M'}
    MACs        - {'ModuleList': '140.73 GMACs'}
    fwd latency - {'ModuleList': '48.48 ms'}
depth 5:
    params      - {'T5LayerFF': '25.17 M'}
    MACs        - {'T5LayerFF': '77.31 GMACs'}
    fwd latency - {'T5LayerSelfAttention': '20.37 ms'}
depth 6:
    params      - {'T5DenseActDense': '25.17 M'}
    MACs        - {'T5DenseActDense': '77.31 GMACs'}
    fwd latency - {'T5Attention': '22.82 ms'}
```

<img width="1421" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/96e5ba68-84b8-43f1-840e-957fdc2a2622">


#### <a name="toc_10"></a>4.1 Training Result (t5-small)


- With batch size of 32, deepspeed splits 5286 training steps into 1764 per epoch for each worker.
```
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
10.254.18.216:   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
10.254.19.151:   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
```

- All 3 worker nodes are consuming the same GPU memory utilization rate consistently at ~5GB:
<img width="1004" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/939a0d56-87e1-4388-bd60-363bff884357">

- NVIDIA® Data Center GPU Manager (DCGM) GPU Utilization metric displayed in Openshift graph: 
<img width="1429" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4ecbe544-8b2f-427b-bc3d-dd6001eefcc9">

- NVIDIA® Data Center GPU Manager (DCGM) Memory Utilization metric displayed in Openshift graph:
<img width="1423" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/017259b4-b716-4243-bc65-1a712b353312">
  
- Time taken by each worker node to complete the training:
```
10.254.21.77: {'train_runtime': 922.0487, 'train_samples_per_second': 183.358, 'train_steps_per_second': 1.913, 'train_loss': 0.23240086172713714, 'epoch': 3.0}
10.254.19.151: {'train_runtime': 922.1271, 'train_samples_per_second': 183.342, 'train_steps_per_second': 1.913, 'train_loss': 0.23220197197531356, 'epoch': 3.0}
10.254.18.216: {'train_runtime': 920.7942, 'train_samples_per_second': 183.608, 'train_steps_per_second': 1.916, 'train_loss': 0.2323370931370188, 'epoch': 3.0}
```

- Tensorboard profiler result:
<img width="1100" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/75680608-acbe-4beb-b5c8-16c8dd1ed376">

#### <a name="toc_10"></a>4.2 Training Result (t5-large)

- All 3 worker nodes are consuming the same GPU memory utilization rate consistently at ~13GB:
<img width="1002" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/97f63ade-170e-47d6-8c0d-54a613e833ac">

- Time taken by each worker node to complete the training:
```
10.254.19.151: {'eval_loss': 0.053917448967695236, 'eval_runtime': 28.1014, 'eval_samples_per_second': 299.664, 'eval_steps_per_second': 3.132, 'epoch': 3.0}
10.254.21.77: {'eval_loss': 0.053917448967695236, 'eval_runtime': 28.1015, 'eval_samples_per_second': 299.664, 'eval_steps_per_second': 3.132, 'epoch': 3.0}
10.254.18.216: {'eval_loss': 0.053917448967695236, 'eval_runtime': 28.0902, 'eval_samples_per_second': 299.784, 'eval_steps_per_second': 3.133, 'epoch': 3.0}
100%|██████████| 1764/1764 [2:55:02<00:00,  5.91s/it]
100%|██████████| 88/88 [00:27<00:00,  3.12it/s]
10.254.21.77: {'train_runtime': 10530.2773, 'train_samples_per_second': 16.055, 'train_steps_per_second': 0.168, 'train_loss': 0.11271674454617663, 'epoch': 3.0}
10.254.19.151: {'train_runtime': 10529.9998, 'train_samples_per_second': 16.056, 'train_steps_per_second': 0.168, 'train_loss': 0.1125946034109241, 'epoch': 3.0}
```

#### <a name="toc_11"></a>4.3 Inference

```
Test Instruction: How many different nationalities do the players of New Jersey Devils come from?
Model Prediction: SELECT COUNT Nationalities FROM FROM table WHERE Players = New Jersey Devils
Expected Answer: SELECT COUNT Nationality FROM table WHERE NHL team = New Jersey Devils
=================================

Test Instruction: What is the nationality of the player from Vancouver Canucks?
Model Prediction: SELECT Nationality FROM table WHERE Player = Vancouver Canucks
Expected Answer: SELECT Nationality FROM table WHERE NHL team = Vancouver Canucks
=================================
Inference took 1.02 seconds
```

### <a name="toc_12"></a>4. deepspeed 2 nodes with ZeRO-1 (t5-large)

#### <a name="toc_13"></a>4.1 Training Result

<img width="1003" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4a510646-beea-47e4-9850-92f8fe8fdd8a">

- With batch size of 32, deepspeed splits 5286 training steps into 2643 per epoch for each worker.
```
  0%|          | 0/2643 [00:00<?, ?it/s]
  0%|          | 0/2643 [00:00<?, ?it/s]
```

#### <a name="toc_14"></a>4.2 Inference


### <a name="toc_9"></a>4. deepspeed 3 nodes with ZeRO-3 Offload (t5-large)

#### <a name="toc_10"></a>4.2 Training Result (t5-large)

<img width="1012" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/f7231d61-8daf-4253-afb0-3345ad81c6c5">






