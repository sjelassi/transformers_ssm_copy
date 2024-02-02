import torch
import numpy as np
from datasets import load_dataset
import json
from qa_evaluation_utils import f1_score, exact_match_score,  metric_max_over_ground_truths
from data_utils import EvalC4CopyDataset, PhoneBookDataset


def copy_c4_evaluation(args,model,tokenizer):

    lengths = np.arange(args.min_eval_len,args.max_eval_len+1)
    print(f"LEN {lengths}")
    str_acc_mean_list = []
    str_acc_std_list  = []
    print("\n")

    for ood_length in lengths:
        str_acc_batch = np.zeros(args.eval_num_batches)
        for jj in range(args.eval_num_batches):

            ##load dataset
            long_dataset = EvalC4CopyDataset(
                    tokenizer, 
                    text_order=args.text_order,
                    batch_size=args.eval_batch_size, 
                    min_length=ood_length, 
                    max_length=ood_length,
                    )
            
            count_questions = 0
            batch = next(iter(long_dataset))
            inputs = batch['input']
            labels = batch['label']
            for (prompt,gt) in zip(inputs,labels):
                tokens = tokenizer(prompt, return_tensors="pt")
                input_ids = tokens.input_ids.to(device="cuda")
                attn_mask = tokens.attention_mask.to(device="cuda")
                max_length = int(1.5*input_ids.shape[1]) + 50
                
                ##generation
                if "mamba" in args.model:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            max_length=max_length,
                            cg=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            enable_timing=False,
                            temperature=0,
                        )
                else:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_length=max_length,
                            return_dict_in_generate=True,
                            temperature=0,
                        )
                out = fn()
                input_length = input_ids.shape[1]
                generated_tokens = out.sequences[:, input_length:]
                
                ##prediction processing
                pred_model=tokenizer.batch_decode(generated_tokens.tolist())[0]
                pred_model = pred_model.split("\n\n")[0]
                pred_model = pred_model.replace("\n"," ").strip()
                
                
                str_acc_batch[jj] += int(gt==pred_model)
                count_questions+=1
                print("\n\n\n")
                print("--"*100,flush=True)
                print(f"CLEAN {pred_model}\n",flush=True)
                print(f"GT {gt}",flush=True)
                print(f"CORRECT {(gt==pred_model)}",flush=True)
                print(f"SEED {jj}; LEN {ood_length}; idx {count_questions}; current result {str_acc_batch[jj]/count_questions}")
                print("--"*100,flush=True)
                print("\n\n\n")
            str_acc_batch[jj] = str_acc_batch[jj]/count_questions

        mean_str_acc = np.mean(str_acc_batch)
        std_str_acc = np.std(str_acc_batch)
        
        str_acc_mean_list.append(mean_str_acc)
        str_acc_std_list.append(std_str_acc)

        print(f"C4 {args.text_order}; len {ood_length}: {mean_str_acc} +- {std_str_acc}")
    
    print("\n")
    return str_acc_mean_list, str_acc_std_list





def phone_book_evaluation(args,model,tokenizer):

    lengths= np.arange(args.min_eval_len, args.max_eval_len+1)

    str_acc_mean_list = []
    str_acc_std_list = []
    for ood_length in lengths:
        str_acc_batch = np.zeros(args.eval_num_batches)
        for jj in range(args.eval_num_batches):
            
            ##load phone book dataset
            long_dataset = PhoneBookDataset(
                    batch_size=args.eval_batch_size,
                    min_length=ood_length,
                    max_length=ood_length
                    )

            batch = next(iter(long_dataset))

            inputs = batch['input']
            labels = batch['label']
            count_questions = 0
            for (prompt,gt) in zip(inputs,labels):
                tokens = tokenizer(prompt, return_tensors="pt")
                input_ids = tokens.input_ids.to(device="cuda")
                attn_mask = tokens.attention_mask.to(device="cuda")
                max_length = input_ids.shape[1] + 50
                
                ##generation
                if "mamba" in args.model:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            max_length=max_length,
                            cg=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            enable_timing=False,
                            temperature=0,
                        )
                else:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_length=max_length,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=0,
                        )
                out = fn()
                
                ##generation post-processing
                input_length = input_ids.shape[1]
                generated_tokens = out.sequences[:, input_length:]
                pred_model=tokenizer.batch_decode(generated_tokens.tolist())[0]
                pred_model = str(pred_model.split("\n")[0].strip())
                str_acc_batch[jj] += int(gt==pred_model)
                print("\n\n\n")
                print("--"*100,flush=True)
                print(f"PHONE-BOOK\n{prompt}\n\n")
                print(f"CLEAN {pred_model}\n",flush=True)
                print(f"GT {gt}",flush=True)
                print(f"CORRECT {(gt==pred_model)}",flush=True)
                print(f"SEED {jj}; LEN {ood_length}; idx {count_questions}; current result {str_acc_batch[jj]/count_questions}")
                print("--"*100,flush=True)
                print("\n\n\n")
                count_questions+=1



        str_acc_batch = str_acc_batch/len(inputs)
        mean_str_acc = np.mean(str_acc_batch)
        std_str_acc = np.std(str_acc_batch)
        
        str_acc_mean_list.append(mean_str_acc)
        str_acc_std_list.append(std_str_acc)
        
        print(f"{args.eval_task}; len {ood_length}: {mean_str_acc} +- {std_str_acc};")

    return str_acc_mean_list, str_acc_std_list


def squad_evaluation(args,model,tokenizer):
    
    filename = "./dir_counter_lengths/squad/dictionary_squad.json" 
    with open(filename) as f_in:
        problems = json.load(f_in)
    
    
    lengths = list(problems.keys())

    print("-"*200,flush=True)
    print(f"LENGTHS {lengths}",flush=True)
    print("-"*200,flush=True)
    num_examples = 30

    em_list = []
    std_em_list = []
    f1_list = []
    std_f1_list = []
    for ood_length in lengths:
        em = 0
        f1 = 0
        tmp_em = []
        tmp_f1 = []
        counter_prob = 0
        examples = problems[ood_length]
        
        for (ctr,context) in enumerate(list(examples.keys())):
            
            qa = examples[context]
            if len(qa)>=2:
                question_1 = qa[0][0]
                ans_1 = qa[0][1]

                question = qa[1][0]
                gt = qa[1][1]

                
                
                prompt = context+"\n\n"+"Question: "+question_1+"\n"+"Answer: "+ans_1[0]+"\n\n"+\
                        "Question: "+question+"\n"+"Answer: "
                tokens = tokenizer(prompt, return_tensors="pt")
                input_ids = tokens.input_ids.to(device="cuda")
                attn_mask = tokens.attention_mask.to(device="cuda")
                max_length = input_ids.shape[1] + 200
                if "mamba" in args.model:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            max_length=max_length,
                            cg=True,
                            return_dict_in_generate=True,
                            output_scores=True,
                            enable_timing=False,
                            temperature=0,
                        )
                else:
                    fn = lambda: model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            max_length=max_length,
                            return_dict_in_generate=True,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=0,
                            begin_suppress_tokens=[tokenizer.eos_token_id],
                        )
                out = fn()
                input_length = input_ids.shape[1]
                generated_tokens = out.sequences[:, input_length:]
                pred_model=tokenizer.batch_decode(generated_tokens.tolist())[0]
                pred_model = str(pred_model.split("\n\n")[0].strip())
                if "<|endoftext|>" in pred_model:
                    pred_model = pred_model.split("<|endoftext|>")[0]
                
                em_for_this_question = metric_max_over_ground_truths(exact_match_score, pred_model, gt)    
                tmp_em.append(em_for_this_question)
                em += int(em_for_this_question)
                f1_for_this_question = metric_max_over_ground_truths(f1_score, pred_model, gt)
                tmp_f1.append(f1_for_this_question)
                f1 += f1_for_this_question
                print("\n")
                print("--"*100)
                print(f"PROMPT\n{prompt}\n\n\n")
                print(f"QUESTION {question}\n\n")
                print(f"\n\nPRED MODEL {pred_model}")
                print(f"\n\n LBL {gt}")
                print(f"LEN {ood_length}; idx {ctr+1}; em {em/(ctr+1)}; f1 {f1/(ctr+1)}",flush=True)
                print("--"*100)
                print("\n")
                counter_prob+=1
                if counter_prob >= num_examples:
                    break
        assert counter_prob >= 15
        f1 = f1/counter_prob#num_examples
        em = em/counter_prob#num_examples

        mean_f1 = np.mean(np.array(tmp_f1))
        std_f1 = np.std(np.array(tmp_f1))

        mean_em = np.mean(np.array(tmp_em))
        std_em = np.std(np.array(tmp_em))

        em_list.append(mean_em)
        f1_list.append(mean_f1)

        std_em_list.append(std_em)
        std_f1_list.append(std_f1)

        print(f"search; {ood_length}; em: {em}; f1: {f1}",flush=True)
    
    print("\n\n\n\n\n")
    print(f"LENGTHS {lengths}",flush=True)
    return em_list, f1_list, std_em_list, std_f1_list






