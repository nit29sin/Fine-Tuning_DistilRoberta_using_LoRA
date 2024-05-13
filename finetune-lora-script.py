import argparse
import time
from functools import partial

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
import torch

from local_dataset_utilities import tokenization, setup_dataloaders, get_dataset
from local_model_utilities import CustomLightningModule
import torch.profiler
from torch.profiler import profile, ProfilerActivity

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.W_a @ self.W_b)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LoRA parameters configuration')
    parser.add_argument('--q_lora', type=str2bool, default=False, help='Apply QLoRA')
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA layers')
    parser.add_argument('--lora_query', type=str2bool, default=True, help='Apply LoRA to query')
    parser.add_argument('--lora_key', type=str2bool, default=False, help='Apply LoRA to key')
    parser.add_argument('--lora_value', type=str2bool, default=True, help='Apply LoRA to value')
    parser.add_argument('--lora_projection', type=str2bool, default=False, help='Apply LoRA to projection')
    parser.add_argument('--lora_mlp', type=str2bool, default=False, help='Apply LoRA to MLP')
    parser.add_argument('--lora_head', type=str2bool, default=False, help='Apply LoRA to head')
    parser.add_argument('--device', type=int, default=0, help='Specify GPU device index')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Enable/disable progress bars')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
        quit()

    df_train, df_val, df_test = get_dataset()
    imdb_tokenized = tokenization()
    train_loader, val_loader, test_loader = setup_dataloaders(imdb_tokenized)

    if args.q_lora:
        nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, quantization_config=nf4_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=args.lora_r, alpha=args.lora_alpha)

    for layer in model.distilbert.transformer.layer:
        if args.lora_query:
            layer.attention.q_lin = assign_lora(layer.attention.q_lin)
        if args.lora_key:
            layer.attention.k_lin = assign_lora(layer.attention.k_lin)
        if args.lora_value:
            layer.attention.v_lin = assign_lora(layer.attention.v_lin)
        if args.lora_projection:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin)
        if args.lora_mlp:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
    if args.lora_head:
        model.pre_classifier = assign_lora(model.pre_classifier)
        model.classifier = assign_lora(model.classifier)

    print("Total number of trainable parameters:", count_parameters(model))

    lightning_model = CustomLightningModule(model)

    callbacks = ModelCheckpoint(
        dirpath='./checkpoints_imdb',     # Directory where the checkpoints will be saved
        filename='checkpoint-{epoch}',  # Name of the checkpoint files, including the epoch                
        save_last=True              # Additionally save the last checkpoint to a separate file
    )

    logger = CSVLogger(save_dir="logs/", name=f"my-model-{args.device}")

    num_epochs = 5
    precision="16-mixed"

    trainer = L.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        accelerator="gpu",
        precision=precision,
        devices=[int(args.device)],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=args.verbose
    )

    start = time.time()

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    print('Memory statistics - ')
    total_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)    
    print("Total memory usage (GB):", total_memory*1e-9)


    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)

    # profiling our model
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True, with_stack=False) as prof:
                 trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)


    print("Profiling has been completed and saved.")    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    prof.export_chrome_trace("./trace.json")

    # Print all argparse settings
    print("------------------------------------------------")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)

    # Print settings and results
    with open("results.txt", "a") as f:
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")        
        for arg in vars(args):
            s = f'{arg}: {getattr(args, arg)}'
            print(s), f.write(s+"\n")

        s = f"Train acc: {train_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Val acc:   {val_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Test acc:  {test_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")  
