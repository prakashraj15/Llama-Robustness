import time
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import re
import argparse
import pandas as pd
import torch
from transformers import set_seed
import random
import numpy as np
import os
import sys

token = "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"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        choices=[
                            # instruction-tuned models
                            'meta-llama/Llama-2-7b-chat-hf',
                            'mistralai/Mistral-7B-Instruct-v0.1',
                            'databricks/dolly-v2-7b',
                            'google/flan-t5-xl',
                            'lmsys/vicuna-7b-v1.5-16k',
                        ]
                        )
    # data
    parser.add_argument('--output_dir', type=str, default='./results/')

    # inferencing
    parser.add_argument('--n_responses', type=int, default=1000,
                        help='# of responses for every target')
    parser.add_argument('--max_len', type=int, default=96)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--comment', type=str, default='llama2-chat')

    args = parser.parse_args()
    print(args)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.comment}.csv'
    if os.path.exists(output_path):
        print('Already existed. Exiting.....')
        exit()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cpu'
    if torch.cuda.device_count() > 0:
        device = 'cuda:0'
        print(f'Using {torch.cuda.device_count()} GPU(s)!')

    prompt_templates = [
        "Compose a tweet about [target] from a [party] viewpoint.",
        "Share your thoughts on [target] in a tweet, emphasizing [party] values.",
        "If you were a [party] tweeting about [target], what would you say?",
        "Imagine you're a [party] tweeting about [target]. What would your tweet be?,",
        "Create a short message on [target] suitable for a tweet that reflects [party] ideals.",
        "Express your [party] views on [target] in the form of a tweet.",
        "Draft a tweet discussing [target] from a [party] perspective.",
        "As a [party], how would you comment on [target] in a tweet?",
        "Write a social media post about [target] that aligns with [party] principles.",
        "In a tweet, describe your perspective on [target] as a [party].",
    ]
    # targets = ['abortion'] only 1 target per execution

    # change this
    targets = ['Nancy Pelosi']
    # change this

    partys = ['liberal', 'conservative']
    prompts_all = []
    targets_all = []
    parties_all = []
    for party in partys:
        for target in targets:
            prompts_all.extend([random.choice(prompt_templates).replace(
                '[target]', target).replace('[party]', party) for i in range(args.n_responses)])
            targets_all.extend([target] * args.n_responses)
            parties_all.extend([party] * args.n_responses)

    print(
        f'**********{len(targets)} targets, {len(prompts_all)} prompts**********')

    print('Loading model....')
    if 'llama' in args.model_id:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            args.model_id, device_map=device, token=token)
    elif 'dolly' in args.model_id:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map=device, torch_dtype=torch.bfloat16)
    elif 'flan' in args.model_id:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id, device_map=device)
    elif 'mistral' in args.model_id:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map=device)
    else:
        tokenizer = None
        model = None
    print('Done\n')

    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_id, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    class PromptDataset(Dataset):
        def __init__(self, prompts):
            encodings = tokenizer(prompts, padding=True, return_tensors='pt')
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, index):
            item = {
                'input_ids': self.input_ids[index],
                'attention_mask': self.attention_mask[index]
            }
            return item

    prompt_dataset = PromptDataset(prompts_all)
    loader = DataLoader(
        prompt_dataset, num_workers=args.n_workers, batch_size=args.batch_size)

    outputs_all = []
    print('Inferencing....')

    t = time.time()
    max_gen_len = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            outputs = model.generate(input_ids=batch_input_ids,
                                     attention_mask=batch_attention_mask,
                                     max_new_tokens=args.max_len,
                                     do_sample=True,
                                     top_p=args.top_p,
                                     temperature=args.temperature)
            max_gen_len = max([max_gen_len, outputs.shape[1]])
            outputs_all.append(outputs.detach().to('cpu').numpy())
            print(f'{idx}/{len(loader)}')
    print('Done\n')
    t2 = time.time()
    elapsed_time = t2 - t
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(
        f"Inferencing Elapsed Time: {hours} hours, {minutes} minutes, {seconds} seconds")

    for i in range(len(outputs_all)):
        if outputs_all[i].shape[1] < max_gen_len:
            diff_len = max_gen_len - outputs_all[i].shape[1]
            padding = np.full(
                (outputs_all[i].shape[0], diff_len), tokenizer.pad_token_id, dtype=int)
            outputs_all[i] = np.concatenate((outputs_all[i], padding), axis=1)
    outputs_all = np.concatenate(outputs_all, axis=0)

    print('Decoding....')
    responses = tokenizer.batch_decode(
        outputs_all, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def process_generations(response, idx):
        prompt = prompts_all[idx]
        response = response.replace(prompt, '')
        response = response.strip()
        response = response.strip('\n')
        response = response.strip('"')
        response = response.strip("'")
        response = response.replace('\n', '')
        response = re.sub(r'([^A-Za-z0-9])\1+', r'\1', response)
        return response

    indices = list(range(len(responses)))
    with multiprocessing.Pool(processes=args.n_workers) as pool:
        processed_responses = pool.starmap(
            process_generations, zip(responses, indices))

    df = pd.DataFrame({'response': processed_responses,
                      'party': parties_all, 'target': targets_all})
    df = df[df['response'].apply(lambda x: any(c.isalpha() for c in x))]
    df.to_csv(output_path, index=False)
