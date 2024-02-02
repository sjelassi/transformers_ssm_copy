import numpy as np
import torch
import string
import torch.nn.functional as F
import random 

class NumberTokenizer:
    def __init__(self, TO_TOKEN, TO_CHAR):
        
        self.TO_TOKEN = TO_TOKEN
        self.TO_CHAR = TO_CHAR

        self.bos_token_id = TO_TOKEN['$']
        self.eos_token_id = TO_TOKEN['.']

    def __call__(self, x):
        encoded = [self.TO_TOKEN[c] for c in x]
        return torch.tensor(encoded, dtype=torch.int64)

    def decode(self, x):
        x = x.detach().cpu().numpy()
        decoded = ''.join([str(t) if t not in self.TO_CHAR else self.TO_CHAR[t] for t in x])
        return decoded

    def __len__(self):
        return len(self.TO_TOKEN)


def arr_to_str(x):
    return ''.join([str(n) for n in x])

def rand_num(length,vocab_size):
        string_ascii_lowercase = string.ascii_lowercase[:vocab_size]
        num = "".join(np.random.choice(list(string_ascii_lowercase),size=length))
        return arr_to_str(num)


def generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size):

    counter_max_ngram = 10
    counter_ngram = 0
    unique = False
    while not unique:
       num1 = rand_num(len1,vocab_size)
       max_limit = len(num1) - n_gram - length_answer -1 if length_answer > 0 else len(num1) - n_gram -1
       list_ngrams = [num1[idx : idx + n_gram] for idx in range(max_limit)]
       unique_n_grams = []
       for ng in list_ngrams:
         if list_ngrams.count(ng) == 1:
           unique_n_grams.append(ng)
       if unique_n_grams:
         unique = True
       counter_ngram +=1
       if counter_ngram >= counter_max_ngram:
          raise ValueError(f"Unable to find a unique {n_gram}-gram in a string of length {len1}!")
    return num1, list_ngrams  


def sample_str(len1,n_gram,length_answer,task,vocab_size=26):

    if task=="copy":
       num1 = rand_num(len1,vocab_size)
       answer = num1[:length_answer] if length_answer > 0 else num1
       example_str = f'${num1}|{answer}.'
    
    elif task == "duplicate_ngram":
       ### sample strings until having one that has a unique n-gram
       num1, list_ngrams = generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size) 
       ngram_new, ngram_old =  random.sample(list_ngrams, 2)

       #create a string with duplicate ngrams
       num1 = num1.replace(ngram_old,ngram_new)

       example_str = f'${num1}|{num1}.'
    elif task in ["prefix_ngram","suffix_ngram"]:
       
       ### sample strings until having one that has a unique n-gram
       num1, list_ngrams = generate_str_unique_ngram(len1, n_gram, length_answer,vocab_size) 

       ngram = list_ngrams[np.random.randint(low=0,high=len(list_ngrams)-1,size=1)[0]]
       index_ngram = num1.index(ngram)

       next_chunk = num1[(index_ngram + len(ngram)):]
       answer = next_chunk[:(length_answer)] if length_answer > 0 else next_chunk

       if task == "prefix_ngram":
          example_str = f'${ngram}|{num1}|{answer}.'
       elif task == "suffix_ngram":
          example_str = f'${num1}|{ngram}{answer}.'
    return example_str 

class CopyDataset:
    def __init__(self, tokenizer, vocab_size=26, n_gram=3, length_answer=-1, train_task="copy", sequence_length=220, min_length=20, max_length=50, num_examples=1000, batch_size=8): 
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.train_task = train_task
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        self.length_answer = length_answer

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'mask': []}
        
        minimal_required_length = self.n_gram if self.n_gram > 0 else 0
        minimal_required_length += self.length_answer if self.length_answer > 0 else 0
        if self.min_length <= minimal_required_length:
            raise ValueError(f"Minimum length is set to {self.min_length} and is smaller than the required one {minimal_required_length}")
        
        minimal_required_length = self.max_length*2
        if self.sequence_length <= minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} because {2*self.max_length}>{self.sequence_length}. Increase your context length !")

        for _ in range(self.batch_size):
            prospective_len = 0
            full_str = ""
            example_mask = []
            while prospective_len < self.sequence_length:
              
              ##sample a string  
              len1 = np.random.randint(self.min_length, self.max_length+1)
              example_str = sample_str(len1,self.n_gram,self.length_answer,self.train_task,self.vocab_size)
              
              ###setting up mask for training loss 
              if self.train_task=="copy":
                 example_mask_tmp = [0] * (len1+2) + [1] * (len(example_str) - len1-2)
              elif self.train_task=="prefix_ngram":
                 example_mask_tmp = [0] * (len1+(self.n_gram+3)) + [1] * (len(example_str) - len1-(self.n_gram+3))
              elif self.train_task == "suffix_ngram":
                 example_mask_tmp = [0] * (len1+self.n_gram+2) + [1] * (len(example_str) - (len1+self.n_gram+2))
              

              #packing the context with examples
              if prospective_len+len(example_str) > self.sequence_length:
                 remaining_len = self.sequence_length - prospective_len
                 remaining_mask_len = self.sequence_length - prospective_len
                 full_str += example_str[:remaining_len]
                 example_mask += [0]*(remaining_mask_len)
                 break
              else:
                 full_str += example_str
                 prospective_len += len(example_str)
                 example_mask += example_mask_tmp

            assert len(full_str) == len(example_mask)
            example_ids = self.tokenizer(full_str)
            example_mask = torch.tensor(example_mask)

            batch['input'].append(full_str)
            batch['input_ids'].append(example_ids)
            batch['mask'].append(example_mask)
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch




class EvalCopyDataset:
    def __init__(self, tokenizer, TO_TOKEN, vocab_size=26, n_gram=3, length_answer=-1, eval_task="copy", sequence_length=220, min_length=8, max_length=30, num_examples=1000, batch_size=8):
        self.min_length = min_length
        self.max_length = max_length
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.TO_TOKEN = TO_TOKEN
        self.vocab_size = vocab_size
        self.eval_task = eval_task
        self.n_gram = n_gram
        self.length_answer = length_answer

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        batch = {'input': [], 'input_ids': [], 'mask': []}
        minimal_required_length = self.max_length*2 
        if self.sequence_length <= minimal_required_length:
            raise ValueError(f"Strings of size {self.max_length} do not fit in a context of size {self.sequence_length} because {2*self.max_length}>{self.sequence_length}. Increase your context length !")

        for _ in range(self.batch_size):
            ##sample a string
            len1 = np.random.randint(self.min_length, self.max_length+1)
            example_str = sample_str(len1,self.n_gram,self.length_answer,self.eval_task,self.vocab_size)
            
            ##fill the context with padding
            example_ids = self.tokenizer(example_str)
            if len(example_ids) < self.sequence_length:
                example_ids = F.pad(example_ids, (0, self.sequence_length-len(example_ids)), value=self.TO_TOKEN['*'])
            

            if self.eval_task=="copy" or self.eval_task=="duplicate_ngram":
                example_mask = torch.tensor([0] * (len1+2) + [1] * (len(example_str) - len1-2) + [0] * (self.sequence_length-len(example_str)))
            elif self.eval_task=="prefix_ngram":
                example_mask = torch.tensor([0] * (len1+self.n_gram+3) + [1] * (len(example_str) - (len1+self.n_gram+3)) + [0] * (self.sequence_length-len(example_str)))
            elif self.eval_task=="suffix_ngram": 
                example_mask = torch.tensor([0] * (len1+self.n_gram+2) + [1] * (len(example_str) - (len1+self.n_gram+2)) + [0] * (self.sequence_length-len(example_str))) 
            
            assert len(example_ids)==len(example_mask)
            batch['input'].append(example_str)
            batch['input_ids'].append(example_ids)
            batch['mask'].append(example_mask)
        batch['input_ids'] = torch.stack(batch['input_ids'], dim=0)
        batch['mask'] = torch.stack(batch['mask'], dim=0)
        return batch






def get_tokenizer(args):
    string_ascii_lowercase = string.ascii_lowercase[:args.vocab_size]
    letters = dict(zip(string_ascii_lowercase, range(args.vocab_size)))

    symbols = {'$': len(letters), '|': len(letters)+1, '.': len(letters)+2, '*': len(letters)+3}

    TO_TOKEN = {**letters, **symbols}

    TO_CHAR = {v:k for k,v in TO_TOKEN.items()}

    tokenizer = NumberTokenizer(TO_TOKEN, TO_CHAR)
    return tokenizer, TO_TOKEN, TO_CHAR


def get_train_dataset(args,tokenizer):

    train_dataset = CopyDataset(
            tokenizer,
            vocab_size=args.vocab_size,
            n_gram=args.n_gram,
            length_answer=args.length_answer,
            train_task=args.train_task,
            sequence_length=args.context_len,
            min_length=args.min_train_len,
            max_length=args.max_train_len,
            batch_size=args.train_batch_size,
            )
    
    return train_dataset


def get_eval_dataset(args, tokenizer, TO_TOKEN, target_min_len,target_max_len):
    
    eval_dataset = EvalCopyDataset(
            tokenizer, 
            TO_TOKEN, 
            vocab_size=args.vocab_size,
            n_gram=args.n_gram,
            length_answer=args.length_answer,
            eval_task=args.eval_task,
            sequence_length=args.context_len,
            min_length=target_min_len, 
            max_length=target_max_len,
            batch_size=args.eval_batch_size,
            )

    return eval_dataset
