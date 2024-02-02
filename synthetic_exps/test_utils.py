import torch
from data_utils import get_eval_dataset
import numpy as np

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_score(args,tokenizer,x,pred,i):
    x_out = tokenizer.decode(x[i])
    x_out = x_out.split('.')[0] + '.'
    pred_out = tokenizer.decode(pred[i])

    if args.eval_task == "prefix_ngram":
        index = find(x_out,'|')[-1]
    elif args.eval_task in ["suffix_ngram","copy","duplicate_ngram"]:
        index = x_out.index('|')


    if args.eval_task=="suffix_ngram":
        gt = x_out[index+1+args.n_gram:][:-1]
        start_idx = index + args.n_gram
    else:
        gt = x_out[index+1:][:-1] 
        start_idx = index

    end_idx = start_idx + len(gt)
    pred_model = pred_out[start_idx:end_idx]
    
    str_acc = int(gt==pred_model) 
    char_acc = sum(map(str.__eq__, gt, pred_model))/max(len(gt),len(pred_model))

    return str_acc, char_acc



def evaluation(args, model, tokenizer, TO_TOKEN):
    

    lengths = np.arange(args.min_eval_len, args.max_eval_len)
    
    str_acc_mean_list = []
    str_acc_std_list = []
    char_accuracy_list = []
    print("\n")
    
    for ood_length in lengths:
        
        str_acc_batch = np.zeros(args.eval_num_batches)
        char_acc_mean = 0

        for jj in range(args.eval_num_batches):

            long_dataset = get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len=ood_length,target_max_len=ood_length)     
            batch = next(iter(long_dataset))
            
            print("-"*100)
            print(f"EXAMPLE {batch['input'][0]}")
            print("-"*100)
            print(batch['input_ids'][-1][batch['mask'][-1]==1], batch['input_ids'][-1], batch['input'][-1])
            print("*"*100)
            
            x = batch['input_ids'].to('cuda')
            
            with torch.no_grad():

                ##prediction
                if args.model=="lstm":
                    state = model.init_hidden(args.eval_batch_size, 'cuda')
                    logits, state = model(x, state)
                elif args.model=="mamba":
                    logits = model(x)[0]
                else:
                    logits = model(x)['logits']
                
                ##greedy decoding
                pred = torch.argmax(logits, dim=-1)


                ##evaluation
                for i in range(len(x)):
                    str_acc, char_acc = get_score(args,tokenizer,x,pred,i) 

                    str_acc_batch[jj] += str_acc
                    char_acc_mean  += char_acc
        

        str_acc_batch = str_acc_batch/len(x)
        mean_str_acc = np.mean(str_acc_batch)
        std_str_acc = np.std(str_acc_batch)

        str_acc_mean_list.append(mean_str_acc)
        str_acc_std_list.append(std_str_acc)

        mean_char_acc = char_acc_mean/(len(x)*args.eval_num_batches)
        char_accuracy_list.append(mean_char_acc)
        
        print(f"{args.eval_task}; len {ood_length}: {mean_str_acc} +- {std_str_acc}; char: {mean_char_acc}")
    print("\n")        
    return str_acc_mean_list, str_acc_std_list, char_accuracy_list



