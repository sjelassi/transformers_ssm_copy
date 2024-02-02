import numpy as np
import torch
import string
import torch.nn.functional as F
import random
from datasets import load_dataset
import names

class EvalC4CopyDataset:
    def __init__(self, tokenizer, text_order="standard", min_length=8, max_length=30, num_examples=1000, batch_size=8):
        
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.dataset = load_dataset("c4","en",split="train[:10%]")
        self.tokenizer = tokenizer
        self.text_order = text_order

    def arr_to_str(self, x):
        return ''.join([str(n) for n in x])

    def rand_num(self, length):
        
        #concatenate 10 texts to ensure having a long enough text
        range_problems = list(range(len(self.dataset)))
        problem_idx = random.sample(range_problems,10)
        example = ""
        for idx_ex in problem_idx:
            example+=self.dataset[idx_ex]["text"]
        tokenized_example = self.tokenizer(example)

        #randomly select a text of length ''length''
        start_position = np.random.randint(0,len(tokenized_example.input_ids)-length,1)[0]
        end_position = start_position + length
        tokenized_text = tokenized_example.input_ids[start_position:end_position]
        # randomly swap order of words if desired
        if self.text_order == "random":
            random_idx = random.sample(range(len(tokenized_text)), len(tokenized_text))
            tokenized_text = [tokenized_text[rr] for rr in random_idx]
            #tokenized_text = tokenized_text[random_idx]
        
        text = self.tokenizer.decode(tokenized_text)
        text = text.replace("\n"," ").strip()
        return text
        
        
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'label': []}
        for _ in range(self.batch_size):

            len1 = np.random.randint(self.min_length, self.max_length+1)
            num1 = self.rand_num(len1)

            first_word = num1.split(" ")[0]
            lbl = " ".join(num1.split(" ")[1:])
            
            ##give first word of text to force it to not output eos
            example_str = f'{num1}\n\n{num1}\n\n{first_word}'
            
            batch['input'].append(example_str)
            batch['label'].append(lbl)
        return batch



class PhoneBookDataset:
    def __init__(self,  min_length=8, max_length=30, num_examples=1000, batch_size=8):
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size

    def arr_to_str(self, x):
        return ''.join([str(n) for n in x])
    
    ##sample a random phone number
    def rand_phone(self,):
        ph_no = []

        # the first number should be in the range of 6 to 9
        ph_no.append(random.randint(6, 9))

        # the for loop is used to append the other 9 numbers.
        # the other 9 numbers can be in the range of 0 to 9.
        for i in range(1, 10):
            ph_no.append(random.randint(0, 9))
        return self.arr_to_str(ph_no)
    
    ##samples a random name + phone number
    def rand_num(self, length):

        num_list = []
        for _ in range(length):
            name = names.get_full_name()
            ph_no = self.rand_phone()
            num_list.append(f"{name}: {ph_no}")

        
        ##randomly select some phone book entries as few shot examples
        idx_fs = random.sample(list(range(len(num_list))), 3)
        few_shot = "\n\n"+num_list[idx_fs[0]]+"\n"+num_list[idx_fs[1]]+"\n\n"
        
        ##prompt
        prompt = "\n".join(num_list)+few_shot
        question = num_list[idx_fs[2]].split(":")[0]+":"
        prompt+= question 
        label = num_list[idx_fs[2]].split(":")[1].strip()
        
        return prompt, label 

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'label': []}
        for _ in range(self.batch_size):

            len1 = np.random.randint(self.min_length, self.max_length+1)
            prompt, label = self.rand_num(len1)
            
            batch['input'].append(prompt)
            batch['label'].append(label)
        return batch








