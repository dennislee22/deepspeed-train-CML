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
[5. deepspeed 2 nodes with 1 GPU each (Zero 2)](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Training Result](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Inference](#toc_11)<br>
[6. deepspeed 2 nodes with 1 GPU each (Zero 3)](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Training Result](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Inference](#toc_14)<br>

### <a name="toc_0"></a>1. Objective

1. When fine-tuning/training a LLM, insufficient VRAM is a major constraint. Major components that will be loaded into the VRAM during training process are:

```
Memory = Model Parameters + Optimiser + Gradient + Activation 
```

2. For instance, training a model of 1 billon parameters with FP32 would require approximately ~22GB of VRAM.

Memory = (4bytes * param) + ((4 bytes/param + 4 bytes/param momentum + 4 bytes/param variance) * param) + (4bytes * param) + <img width="363" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/4c647806-3634-437b-aba4-d7581437aa59">
 
3. Thus, training a 1B or 7B model with substantial amount of dataset might be able to fit into a single GPU device with 40GB of memory and the latter might need to involve quantization technique when the training takes place. So the question is how to train a bigger model beyond 7B parameters with 40GB GPU cards. Techniques include:
&nbsp;a.Pipeline Par

6. The provided iPython codes in this repository serve as a comprehensive illustration of the complete lifecycle for fine-tuning a particular Transformers-based model using specific datasets. This includes merging LLM with the trained adapters, quantization, and, ultimately, conducting inferences with the correct prompt. The outcomes of these experiments are detailed in the following section. The target use case of the experiments is making use the Text-to-SQL dataset to train the model, enabling the translation of plain English into SQL query statements.<br>
&nbsp;a. [ft-trl-train.ipynb](ft-trl-train.ipynb): Run the code cell-by-cell interactively to fine-tune the base model with local dataset using TRL (Transformer Reinforcement Learning) mechanism. Merge the trained adapters with the base model. Subsequently, perform model inference to validate the results.<br>
&nbsp;b. [quantize_model.ipynb](ft-trl-train.ipynb): Quantize the model (post-training) in 8, or even 2 bits using `auto-gptq` library.<br>
&nbsp;c. [infer_Qmodel.ipynb](ft-trl-train.ipynb): Run inference on the quantized model to validate the results.<br>
&nbsp;d. [gradio_infer.ipynb](gradio_infer.ipynb): You may use this custom Gradio interface to compare the inference results between the base and fine-tuned model.<br>
7. The experiments also showcase the post-quantization outcome. Quantization allows model to be loaded into VRAM with constrained capacity. `GPTQ` is a post-training method to transform the fine-tuned model into a smaller footprint. According to [🤗 leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), quantized model is able to infer without significant results degradation based on the scoring standards such as MMLU and HellaSwag. `BitsAndBytes` (zero-shot) helps further by applying 8-bit or even 4-bit quantization to model in the VRAM to facilitate model training. 
8. Experiments were carried out using `bloom`, `falcon` and `codegen2` models with 1B to 7B parameters. The idea is to find out the actual GPU memory consumption when carrying out specific task in the above PEFT fine-tuning lifecycle. Results are detailed in the following section. These results can also serve as the GPU buying guide to achieve a specific LLM use case.
 
#### <a name="toc_1"></a>2. Summary & Benchmark Score

- Graph below depicts the GPU memory utilization during a specific stage. This graph is computed based on the results obtained from the experiments as detailed in the tables below.

<img width="897" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/c50dbbcc-41a5-4f51-a09d-ac27b21dcb58"><br>

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

&nbsp;&nbsp;b. Time taken to quantize the fine-tuned (merged with PEFT adapters) model using `auto-GPTQ` technique:<br>

| Model      | Quantization Technique| Quantization Duration | Inference Result  |
| :---       |     :---:           |   ---:                  | :---              |
| bloom-1b1  | auto-gptq 8-bit     | ~5 mins                 | Bad               |
| bloom-7b1  | auto-gptq 8-bit     | ~35 mins                | Good              |
| falcon-7b  | auto-gptq 8-bit     | ~22 mins                | Good              |

&nbsp;&nbsp;c. Table below shows the amount of memory of a A100-PCIE-40GB GPU utilised during specific experiment stage with different models.

### <a name="toc_2"></a>3. Preparation

#### <a name="toc_3"></a>3.1 Build Custom Docker Image

- Build a Docker image locally (based on the CML image with Jupyter notebook) and push it to the external docker registry, represented by Nexus repository in this example.

```
docker build -t dlee-deepspeed:2024.1.1 . -f deepspeed-pdsh-mpi-nvcc-jupyter
docker tag dlee-deepspeed:2024.1.1 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jupyter:2024.1.1
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-jupyter:2024.1.1
```

- Build another Docker image locally (based on the CML image with Workbench notebook) and push it to the external docker registry.

```
docker build -t dlee-deepspeed:2024.1.1 . -f deepspeed-pdsh-mpi-nvcc-wb
docker tag dlee-deepspeed:2024.1.1 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.1
docker push 10.113.204.134:9999/pvcds152/p3.10-nvcc-pdsh-mpi-wb:2024.1.1
```
#### <a name="toc_4"></a>3.2 Dataset & Model

#### <a name="toc_5"></a>3.5 CML Session


### <a name="toc_6"></a>4. Single node with 1 GPU

#### <a name="toc_7"></a>4.1 Training Result

<img width="983" alt="image" src="https://github.com/dennislee22/deepspeed-train-CML/assets/35444414/b027bd5c-81f2-4406-b1bc-d7fe94775ecf">

#### <a name="toc_8"></a>4.2 Inference

### <a name="toc_9"></a>4. deepspeed 2 nodes with 1 GPU each (Zero 2)

#### <a name="toc_10"></a>4.1 Training Result

#### <a name="toc_11"></a>4.2 Inference

### <a name="toc_12"></a>4. deepspeed 2 nodes with 1 GPU each (Zero 3)

#### <a name="toc_13"></a>4.1 Training Result

#### <a name="toc_14"></a>4.2 Inference

- Tables below summarize the benchmark result when running the experiments using 1 unit of Nvidia A100-PCIE-40GB GPU on CML with Openshift (bare-metal):<br>





