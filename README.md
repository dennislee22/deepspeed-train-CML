# deepspeed Training in <img width="36" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/38692491-f56b-4fc0-a40e-8412b35327b4"> CML 

<img width="407" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/df7fb23c-8828-479c-9a5a-a49fa1969be0">

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Benchmark Score & Summary](#toc_1)<br>
[3. Preparation](#toc_2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Build Custom Docker Image](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. Create Tensorboard in CML Application](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3. Create CML Session](#toc_5)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.4. Prepare Dataset & Model](#toc_6)<br>
[4. Single Node/Pod without ZeRO](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Training Result without ZeRO (t5-small)](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Training Result without ZeRO (t5-large)](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.3. Inference](#toc_10)<br>
[5. deepspeed 3 Nodes/Pods with ZeRO-1](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Training Result with ZeRO-1 (t5-small)](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Training Result with ZeRO-1 (t5-large)](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.3. Inference](#toc_14)<br>
[6. deepspeed 2 Nodes/Pods with ZeRO-1](#toc_15)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Training Result with ZeRO-1 (t5-large)](#toc_16)<br>
[7. deepspeed 3 Nodes/Pods with ZeRO-3 Offload](#toc_17)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Training Result with ZeRO-3 Offload (t5-large)](#toc_18)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.2. Inference](#toc_19)<br>

### <a name="toc_0"></a>1. Objective

- When fine-tuning/training a LLM, insufficient VRAM is a major constraint. First, let's understand the actual GPU memory requirements for fine-tuning a model. 
- In general, the major components that will be loaded into the VRAM during LLM training process are as follows.

```
VRAM (training/fine-tuning) = Model Parameters + Optimizer + Gradient + Activation 
```

- For instance, training a model of 1 billon parameters with FP32 would require approximately ~22GB of VRAM.

VRAM (training/fine-tuning) =<br>
<sup>(4bytes * param) + ((4 bytes/param + 4 bytes/param momentum + 4 bytes/param variance) * param) + (4bytes * param) + </sup><img width="300" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4c647806-3634-437b-aba4-d7581437aa59">
 
- Thus, training a 1B or 7B model with substantial amount of dataset might be able to fit into a single GPU device with 40GB of memory and the latter might need to involve quantization technique when the training takes place. So, the question is how to train a bigger model with billions of parameters given the limited VRAM size. The available techniques today include Pipeline Parallelism (PP), Data Parallelism (DP) and Tensor Parallelism (TP) or even 3D Parallelism.
- This article focuses on fine-tuning LLM suing **[ZeRO](https://github.com/microsoft/DeepSpeed)** Redundancy Optimizer (ZeRO) technique. ZeRO is capable of sharding the 3 model states (optimizer states, gradients, and parameters) across data-parallel processes instead of replicating them (as practised by DP). In other words, **ZeRO allows fitting huge LLM into the GPUs with restricted memory, both intra-node and inter-node.**

#### <a name="toc_1"></a>2. Summary & Benchmark Score

- The target use case of the experiments is fine-tuning the model with Text-to-SQL dataset with/without ZeRO, enabling the translation of plain English into SQL query statements. Experiments were carried out using `t5-small` and `t5-large` models with 60 million and 770 million parameters respectively in CML v1.5.2 running on Openshift platform.
- The experiments utilize `batch size=32` configuration for fine-tuning/training the models. Although using higher batch size would increase the training speed, batch size 32 is selected to perform apple-to-apple comparison of the training outcome without/with ZeRO technique in place.
- As `t5-large` model has [issue](https://discuss.huggingface.co/t/t5-variants-return-training-loss-0-and-validation-loss-nan-while-fine-tuning/30839) with FP16 during training, FP32 is configured for the experiments. 
- Table below summarizes the benchmark outcome as the result of running the experiments. Each running pod is attached to 1 unit of Nvidia A100-PCIE-40GB device.

| Model     | Technique           | Total Node/Pod | Duration | Inference Result    | Memory (each Pod)  |
| :---      |     :---:           |  :---:         |  ---:   |   :---:             |   :---:            |
| t5-small  | w/o deepspeed       |     1          | ~742 secs | Good                |   3 GB             |
| t5-large  | w/o deepspeed       |     1          | ~ | Good                |   15 GB            |
| t5-small  | deepspeed ZeRO-1    |     3          | ~922 secs | Good                |   5 GB             |
| t5-large  | deepspeed ZeRO-1    |     3          | ~10530 secs | Good                |   13 GB            |
| t5-large  | deepspeed ZeRO-1    |     2          |          | Good                |   15 GB            |
| t5-large  | deepspeed ZeRO-3 Offload  |     3    | ~11044 secs | Good                |   9 GB             |

#### Summary:
- deepspeed `ZeRO-1` with 3 nodes/pods manage to reduce the VRAM consumption when training `t5-large` model, but at the expense of slower training speed compared to single node/pod training without deepspeed.
-  When training LLM in the multi-nodes landscape, the speed is often bottlenecked by network communication overhead (both physical underlay and virtual overlay network) and GPU-CPU-GPU transition process. This can be overcome by resorting to compelling options such as SR-IOV and Infiniband technology. Here's the [reference](https://docs.nvidia.com/networking/display/public/sol/rdg+for+accelerating+ai+workloads+in+red+hat+ocp+with+nvidia+dgx+a100+servers+and+nvidia+infiniband+fabric#src-99399137_RDGforAcceleratingAIWorkloadsinRedHatOCPwithNVIDIADGXA100ServersandNVIDIAInfiniBandFabric-OpenShiftContainerPlatformNetworking).
- deepspeed `ZeRO-3 Offload` can exploit both GPU and CPU memory in order to optimize VRAM consumption further compared to `ZeRO-1`. Optimizers such as Adam, can consume a significant amount of GPU compute and memory. ZeRO-Offload reduces the GPU compute and memory requirements of such models by leveraging compute and memory resources on the host CPU to execute the optimizer. Furthermore, to prevent the optimizer from becoming a bottleneck, ZeRO-Offload uses DeepSpeed’s highly optimized CPU implementation of Adam called DeepSpeedCPUAdam. DeepSpeedCPUAdam is 5X–7X faster than the standard PyTorch implementation.
- The model size must be significantly huge to take advantage of the deepspeed technology. As seen in `t5-small` model training result, the loaded VRAM is lower than with deepspeed.
 

### <a name="toc_2"></a>3. Preparation

- The LLM training in the following experiments use 🤗 Transformers and PyTorch software packages. PyTorch 2.1.2 requires CUDA12.1 as shown below.  
<img width="425" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/d739357e-1421-439d-9395-2bbdf03bbd57">
- This article uses docker image installed with [Nvida CUDA nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) version 12.2 for fixing some other incompatibilities.
- As a reference, the outcome of the experiments shows that CUDA nvcc 12.2 can be used as reported in the following training log.
```
Installed CUDA version 12.2 does not match the version torch was compiled with 12.1 but since the APIs are compatible, accepting this combination
```

#### <a name="toc_3"></a>3.1 Build Custom Docker Image

- Build a Docker image locally (based on the native CML image with Jupyter notebook) and push it to the external docker registry, which is represented by Nexus repository in this example.
- The image is installed with the required Nvidia packages. Specific CUDA packages can be referenced from this [Nvidia (ubuntu2004 image)](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/) site.
- For inter-nodes training deployment, deepspeed uses launchers such as OpenMPI and PDSH (a variant of rsh) which are both installed in the docker image as well.

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-jupyter
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jptr:2024.1.1
```

- Build another Docker image locally (based on the CML image with Workbench notebook) and push it to the external docker registry. Use this image instead of iPython, if you want to run the training process in the form CML job.

```
docker build -t dlee-deepspeed:2024.1.4 . -f deepspeed-pdsh-mpi-nvcc-wb
docker tag dlee-deepspeed:2024.1.4 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.4
```

- Register the new image in CML.

<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/38c82e3c-2ee4-4e00-9fb1-7a2f2c582779"><br>

- Verify that the image has been registered succesfully.

<img width="500" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/bdc45baa-54a2-4e39-afa1-7e4ff8988192"><br>

#### <a name="toc_4"></a>3.2 Create CML Session

- Create a new CML project with Python 3.10 and GPU variant.

- Add the newly registered image in the CML project.

<img width="1422" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/a88ca709-a10b-43f1-bd30-b9f6786bafbc"><br>

- Add the following environment variables in the CML project.

<img width="1185" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/299b4736-b9fc-4f91-9f8f-09e52bd25f5d"><br>

- Create a new CML session in the project.
  
<img width="1414" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/0ab49111-1b91-4491-9e81-605822a7f84d"><br>

- Open the Terminal window in the CML session and run the following commands to replace the CUDA path with the installed version in the custom docker image.
  
```
$ rm /usr/local/cuda
$ ln -s /usr/local/cuda-12.2 /usr/local/cuda
$ ls -l /usr/local/cuda
lrwxrwxrwx. 1 cdsw cdsw 20 Jan  4 05:38 /usr/local/cuda -> /usr/local/cuda-12.2
```
- Install the Python packages.

```
pip install -r requirements.txt
```

- Verify the status of deepspeed.

<img width="500" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/abe5a96d-780c-4fe7-b8aa-f943317ec3ff"><br>


#### <a name="toc_5"></a>3.3 Create Tensorboard in CML Application

- Tensorboard is deployed to monitor the training/validation loss. The training script will report to Tensorboard.
- Create Tensorboard in the CML application
<img width="476" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/f7a42bef-9c1e-4910-a68b-b9b9961ba831">

- Upon successful creation, browse the Tensorboard website.
<img width="571" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/68b4c50e-b536-458e-ad00-7b67716097af">


#### <a name="toc_6"></a>3.4 Prepare Dataset & Model

- In the CML session, run the [prep_dataset.ipynb](prep_dataset.ipynb) to prepare/tokenize the wikiSQL dataset prior to fine-tuning the model.
- In the CML session, you may opt to clone the LFS model in advance.

```
git-lfs clone
```

### <a name="toc_7"></a>4. Single Node/Pod without ZeRO

- Train the cloned `t5-small` model with the tokenized dataset using [textsql_train.py](textsql_train.py) script. The default value of other parameters can be changed/added in the argument list if necessary. Please explore the script for more information.
```
!python textsql_train.py \
--model_id 't5-small' \
--outputdir small-trainoutput-no_ds \
--epochs 3 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_checkpointing False
```

#### <a name="toc_8"></a>4.1 Training Result without ZeRO (t5-small)

- The single node/pod consumes the GPU memory consistently throughout the training process at ~3GB:

<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/17fba932-61a7-4653-9159-bf2f73ace7b4"><br>

- Time taken by single node/pod to complete the training:
```
{'train_runtime': 742.5369, 'train_samples_per_second': 227.686, 'train_steps_per_second': 7.119, 'train_loss': 0.16859441952772136, 'epoch': 3.0}
```

- Tensorboard Profiler (Training + Validation Loss combined):
<img width="700" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/65ed4421-0ca3-456b-a62a-b4f5806be69b"><br>

#### <a name="toc_9"></a>4.1 Training Result without ZeRO (t5-large)

- The single node/pod consumes the GPU memory consistently throughout the training process at ~15GB:
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/c7723691-eee7-4b4f-a245-bf151c87a148"><br>

#### <a name="toc_10"></a>4.2 Inference

- Run [run_inference.ipynb](run_inference.ipynb) for model inference and check the results.
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

### <a name="toc_11"></a>5. deepspeed 3 Nodes/Pods with ZeRO-1

- Run [deepspeed-train.ipynb](deepspeed-train.ipynb) script to fine-tune the model using deepspeed technique. The first cell is designed to launch the necessary CML worker pods. The CML worker pods use the same image as the current CML session which has the necessary Nvidia software packages, pdsh/openMPI and openSSH installed. In this example, deepspeed uses pdsh with SSH protocol to run the training script in the remote worker pods.

```
from cml.workers_v1 import launch_workers
import subprocess, socket, os, sys
from subprocess import call 

NUM_WORKERS = 3
worker_cpu = 4
worker_memory = 32
worker_gpu = 1
hostfile = "/home/cdsw/hostfile.txt"

def display_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        print(content)

#preparing hostfile with pdsh/openMPI specific format
def redirect_to_file(text):
    file = open(hostfile, 'a')
    file.write(text + " slots=1\n")
    file.close
    
for i in range(NUM_WORKERS):
    worker_cmd = "python worker_p.py"
    print(f"Launch CML worker pod {i}...")
    # worker0 runs a different script
    if i == 0:
        with open('/home/cdsw/hostfile.txt', 'w') as f_obj:
            call(['python', 'master_p.py'], stdout=f_obj)
    else:
        launch_workers(name=f'CML Worker Pods {i}', n=1, cpu=worker_cpu, memory=worker_memory, nvidia_gpu = worker_gpu,  code="!"+worker_cmd )
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("0.0.0.0", 6000))
        s.listen(1)    
        conn, addr = s.accept()
        for i in range(2):
            data = conn.recv(20)
            if not data: break
            string = str(data, encoding='utf-8')
            redirect_to_file(string)
            conn.send("Hello From Server!".encode())
        conn.close()

print("Content of hostfile:")
display_file_content(hostfile)
```
```
Launch CML worker pod 0...
Launch CML worker pod 1...
Launch CML worker pod 2...
Content of hostfile:
10.254.21.79 slots=1
10.254.18.217 slots=1
10.254.19.152 slots=1
```

- Run the following cell to execute deepspeed training script.
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

- DeepSpeed Flops Profiler (`zprofiler_result.txt` as defined in the `dsconfig/zero1profiler.json` file):
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

<img width="900" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/96e5ba68-84b8-43f1-840e-957fdc2a2622">


#### <a name="toc_12"></a>5.1 Training Result with ZeRO-1 (t5-small)

- With batch size of 32, deepspeed splits 5286 training steps into 1764 per epoch for each worker.
```
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
10.254.18.216:   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
10.254.19.151:   total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
  0%|          | 0/1764 [00:00<?, ?it/s]/home/cdsw/.local/lib/python3.10/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1652: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
```

- All 3 worker nodes/pods are consuming the same amount of GPU memory consistently throughout the training process at ~5GB:
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/939a0d56-87e1-4388-bd60-363bff884357">

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
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/75680608-acbe-4beb-b5c8-16c8dd1ed376">

#### <a name="toc_13"></a>5.2 Training Result with ZeRO-1 (t5-large)

- All 3 worker nodes/pods are consuming the same amount of GPU memory consistently throughout the training process at ~13GB:
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/97f63ade-170e-47d6-8c0d-54a613e833ac">

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

#### <a name="toc_14"></a>5.3 Inference

- Execute [run_inference.ipynb](run_inference.ipynb) to load the fine-tuned model for inference and check the results.
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

### <a name="toc_15"></a>6. deepspeed 2 Nodes/Pods with ZeRO-1

#### <a name="toc_16"></a>6.1 Training Result with ZeRO-1 (t5-large)

- All 2 worker nodes/pods are consuming the same amount of GPU memory consistently throughout the training process at ~14GB:
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4a510646-beea-47e4-9850-92f8fe8fdd8a">

- With batch size of 32, deepspeed splits 5286 training steps into 2643 per epoch for each worker.
```
  0%|          | 0/2643 [00:00<?, ?it/s]
  0%|          | 0/2643 [00:00<?, ?it/s]
```

### <a name="toc_17"></a>7. deepspeed 3 Nodes/Pods with ZeRO-3 Offload

- Training script is as follows:
```
!export PDSH_SSH_ARGS_APPEND='';deepspeed --hostfile /home/cdsw/hostfile.txt \
--launcher pdsh \
--num_nodes 3 \
--num_gpus 1 \
--master_addr $CDSW_IP_ADDRESS \
--ssh_port 2222 textsql_train.py \
--model_id 't5-large' \
--outputdir ds-train-zero3-large \
--epochs 3 \
--gradient_checkpointing False \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--deepspeed dsconfig/zero3profiler.json
```

#### <a name="toc_18"></a>7.1 Training Result with ZeRO-3 Offload (t5-large)

- All 3 worker nodes/pods are consuming the same amount of GPU memory consistently throughout the training process at ~9GB:
<img width="800" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/f7231d61-8daf-4253-afb0-3345ad81c6c5">

- Time taken by each worker node to complete the training:
```
10.254.18.217: {'train_runtime': 11044.5628, 'train_samples_per_second': 15.308, 'train_steps_per_second': 0.16, 'train_loss': 0.11009906154641219, 'epoch': 3.0}
10.254.19.152: {'train_runtime': 11044.5643, 'train_samples_per_second': 15.308, 'train_steps_per_second': 0.16, 'train_loss': 0.10998346529850343, 'epoch': 3.0}
10.254.21.79: {'train_runtime': 11044.3776, 'train_samples_per_second': 15.308, 'train_steps_per_second': 0.16, 'train_loss': 0.11003240544239139, 'epoch': 3.0}
```

#### <a name="toc_19"></a>7.2 Inference

- Execute [run_inference.ipynb](run_inference.ipynb) to load the fine-tuned model for inference and check the results.
```
Test Instruction: What college did Calvin McCarty play at?
Model Prediction: SELECT College FROM table WHERE College = Calvin McCarty
Expected Answer: SELECT College FROM table WHERE Player = Calvin McCarty
=================================

Test Instruction: What is the composition at Valles lava dome?
Model Prediction: SELECT composition FROM table WHERE composition = Valles Lava dome
Expected Answer: SELECT Composition FROM table WHERE Name of lava dome = Valles lava dome
=================================

Test Instruction: What song has a length of 3:05?
Model Prediction: SELECT Song FROM table WHERE Duration = 3:05
Expected Answer: SELECT Song FROM table WHERE Length = 3:05
=================================

Inference took 1.06 seconds
```




