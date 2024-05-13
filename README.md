# finetune DistilBert using LORA

<h3> Project Description </h3>
Goal is to fine-tune an LLM using Low Rand Adaptation (LoRA) and compare its performance, efficiency, and accuracy with traditional fine-tuning along with exploring the effects of varying optimizers, learning rates, dataset and applying quantization (QLoRA). <br>
With this we aspire to refine the balance between computational efficiency and model performance. <br> 
This endeavour will explore the compute-memory trade-off, aiming to establish a set of best practices for developing highly efficient, accurate Large Language models. 




<h3> A description of the repository and code structure </h3>

finetune-lora.py -> This is the main file to start with.  <br>
local_dataset_utilities.py and local_model_utilities.py are utility files. <br>

DistilRoberta directory contains the code of our attempt to finetune DistilRoberta model on Financial Sentiment dataset.

<h3> Example commands to execute the code        </h3>

This code needs to be run on GPU.  <br>

To run on a GPU enabled system, run finetune-lora.py with appropriate command line arguments. <br>

To run on HPC - <br>

run following batch file - run_DistilBert_gpu.SBATCH with appropriate command line arguments <br>

following is the list of command line arguments - <br>

--q_lora (boolean) -> To enable QLoRA or not <br>
--lora_r (int)     -> Rank for LoRA layers <br>
--lora_alpha (int)  -> Alpha for LoRA layers <br>
--lora_query (boolean) ->  Apply LoRA to query <br>
--lora_key  (boolean)  ->  Apply LoRA to key <br>
--lora_value (boolean) -> Apply LoRA to value <br>
--lora_projection (boolean) -> Apply LoRA to linear projection layer <br>
--lora_mlp (booealn) -> Apply LoRA to linear MLP layer <br>
--lora_head (boolean) -> Apply LoRA to linear attention head <br>



<h3> Results  </h3>

<ul>
<li> Finetuning using LoRA (r=8,alpha=16), Test Accuracy - 89.71%</li>
<li> Finetuning without LoRA (selective finetuning), Test Accuracy - 87.39%</li>
<li> Finetuning without LoRA (full finetuning), Test Accuracy - 91.59%</li>
<li> QLoRA, Test Accuracy - 91.99%</li>
<li> LoRA with ADAM, Test Accuracy - 92.39%</li>
<li> LoRA with SGD, Test Accuracy - 66.45%</li>
<li> LoRA with SGD and Nesterov, Test Accuracy - 91.46%</li>
<li> LoRA with LinearLR Schedular, Test Accuracy - 92.44%</li>
<li> LoRA with OneCycleLR Schedular, Test Accuracy - 50.06%</li>

</ul>
