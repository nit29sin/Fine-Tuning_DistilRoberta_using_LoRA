# finetune DistilBert using LORA

<h3> Project Description </h3>
Goal is to fine-tune an LLM using Low Rand Adaptation (LoRA) and compare its performance, efficiency, and accuracy with traditional fine-tuning along with exploring the effects of varying optimizers, learning rates, dataset and applying quantization (QLoRA). <br>
With this we aspire to refine the balance between computational efficiency and model performance. <br> 
This endeavour will explore the compute-memory trade-off, aiming to establish a set of best practices for developing highly efficient, accurate Large Language models. 


<h3> Project milestones and their completion status </h3>

<ul>
<li>We integrated LoRA into the linear layers of the transformer's distilBert model by incorporating low-rank matrices. - <b>DONE</b> <br> </li>
<li>We did an exhaustive grid search on LoRA parameters and linear layers to find the best parameters and best combination of layers for optimal results. - <b>DONE</b> <br> </li>
<li>We did a full fine-tuning and selective fine-tuning to compare its accuracy and efficiency with LoRA on IMDB dataset using DistilBert model. - <b>DONE</b> <br> </li>
<li>With optimal LoRA configuration, we found out the effect of applying quantization (QLoRA) by converting our base model to nf4 precision before applying LoRA weights. - <b>DONE</b> <br> </li>
<li>To find out the effect of optimizer, we ran LoRA with ADAM, SGD and SGD with Nesterov. - <b>DONE</b> <br> </li>
<li>We also experimented by varying learning rate using learning rate schedular. - <b>DONE</b> <br> </li>
<li>To find out the effect on dataset and model, we added LoRA on a smaller model (DistilRoberta) and ran on a smaller dataset (hugging face financial sentiment). - <b>DONE</b> <br> </li>
</ul>

<h3> A description of the repository and code structure </h3>

finetune-lora.py -> This is the main file to start with.  <br>
local_dataset_utilities.py and local_model_utilities.py are utility files. <br>
gridsearch.py -> This file contains the code for grid search <br>
grid_search_results.txt -> This file contains the result of grid search <br>

DistilRoberta directory contains the code of our attempt to finetune DistilRoberta model on Financial Sentiment dataset.

<h3> Example commands to execute the code        </h3>

This code needs to be run on GPU.  <br>

To run on a GPU enabled system, run finetune-lora.py with appropriate command line arguments. <br>

To run on HPC - <br>

run following batch file - run_DistilBert_gpu.SBATCH with appropriate command line arguments <br>

following is the list of command line arguments - <br>

--q_lora boolean -> To enable QLoRA or not <br>
--lora_r int     -> Rank for LoRA layers <br>
--lora_alpha int  -> Alpha for LoRA layers <br>
--lora_query boolean ->  Apply LoRA to query <br>
--lora_key  boolean  ->  Apply LoRA to key <br>
--lora_value boolean -> Apply LoRA to value <br>
--lora_projection boolean -> Apply LoRA to projection layer <br>
--lora_mlp booealn -> Apply LoRA to MLP <br>
--lora_head boolean -> Apply LoRA to head <br>



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
