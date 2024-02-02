from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm
from pathlib import Path
from torch.optim import AdamW
import torch
import os

def ce_loss(inputs, logits, mask, TO_TOKEN):
    # Shift so that tokens < n predict n
    if type(logits) != torch.Tensor:
        logits = logits['logits']
    shift_labels = inputs.contiguous()
    shift_logits = logits.contiguous()
    mask = mask.contiguous().view(-1)

    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(ignore_index=TO_TOKEN['*'], reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.sum(loss*mask)/torch.sum(mask)


def get_optimizer(model,args):
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    return optimizer

def custom_get_scheduler(optimizer,num_training_steps):

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def train(args,model, train_dataset, TO_TOKEN):
    
    optimizer = get_optimizer(model,args)
    
    ## Set model to GPU
    from accelerate import Accelerator

    accelerator = Accelerator()

    model, optimizer = accelerator.prepare(
        model, optimizer
    )


    num_train_epochs = args.epochs
    num_update_steps_per_epoch = args.steps
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    num_log_steps = 50

    lr_scheduler = custom_get_scheduler(optimizer,num_training_steps)
    


    gradient_accumulation_steps = 1

    model.train()
    completed_steps = 0
    num_train_epochs = 1

    for epoch in range(num_train_epochs):
        avg_loss = [0]
        count = [0]
        progress_bar = tqdm(
            enumerate(train_dataset, start=1), total=num_training_steps,
            desc=f'Epoch {epoch + 1}/{num_train_epochs}'
        )
        for step, batch in progress_bar:
            x = batch['input_ids'][:,:-1].to('cuda')
            y = batch['input_ids'][:,1:].to('cuda')
            mask = batch['mask'][:,1:].to('cuda')

            if args.model=="lstm":
                state = model.init_hidden(args.train_batch_size, 'cuda')

            if args.model=="lstm":
               logits, state = model(x, state)
            else:
               logits = model(x)

            if args.model=="mamba":
                logits = logits[0]


            loss = ce_loss(y, logits, mask, TO_TOKEN)
            if (step+1) % num_log_steps == 0:
                avg_loss.append(0)
                count.append(0)
            loss = loss / gradient_accumulation_steps
            avg_loss[-1] += loss.item()
            count[-1] += 1
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if step > num_training_steps:
                break
            # Update tqdm description with the current loss
            progress_bar.set_postfix({'Loss': loss.item()})



def save_model(args, model):


    if args.model.startswith("T"):

        save_path = "./output_dir/"+f"model_{args.model}_layer_{args.layers}_hidden_{args.hidden_size}_heads_{args.heads}_train_{args.train_task}_lr_{args.lr}_epochs_{args.epochs}_steps_{args.steps}/"


    elif args.model == "lstm" or args.model == "mamba":

        save_path = "./output_dir/"+f"model_{args.model}_layer_{args.layers}_hidden_{args.hidden_size}_train_{args.train_task}_lr_{args.lr}_epochs_{args.epochs}_steps_{args.steps}/"

    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)

    #save model
    if args.model=="lstm" or args.model=="mamba":
        save_path += "model.pt"
        torch.save(model,save_path)
    else:
        model.save_pretrained(save_path)
