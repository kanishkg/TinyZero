import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='openwebmath_none', help='Dataset name')
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--user', type=str, default='Asap7772', help='User to push the dataset to')

def get_prompts(ds, tokenizer, prompt_templates):
    prompts = []
    tokenized_inputs = tokenizer(ds['text'])
    samples = []
    max_seq_length = 4096
    for e, example in tqdm(enumerate(tokenized_inputs['input_ids']), desc="Truncating prompts"):
        if len(example) > max_seq_length-1024:
            sample = tokenizer.decode(example[: max_seq_length - 1024])
            sample = sample[: sample.rfind("\n")]
            samples += [sample]
        else:
            samples += [ds['text'][e]]

    for example in tqdm(samples, desc="Generating prompts"):
        prompt = prompt_templates['qa_none'] + f"\n<text>\n{example}\n</text>"
        prompt = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': ''}]
        prompts += [prompt]
  
    new_prompts = [tokenizer.apply_chat_template(
        p,
        tokenize=False,
    ) for p in prompts]
    return new_prompts

def parse_output(output):
    query_match = re.search(r'<question>(.*?)</question>', output, re.DOTALL)
    think_match = re.search(r'<thoughts>(.*?)</thoughts>', output, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    
    query = query_match.group(1) if query_match else ""
    think = think_match.group(1) if think_match else ""
    answer = answer_match.group(1) if answer_match else ""
    
    completion = f"<think>{think}</think>\n<answer>{answer}</answer>" if think or answer else ""
    return query, completion

def main(args):
    prompt_templates = {
        'qa': """Your goal is to split the text into a question, thought and an answer.

Make sure that the question is in the text. 
Make sure that the answer and the process of the writer to get to the answer are in the text. Do not change the wording of either.
Your goal is to copy and paste the question, the process and the answer into the correct sections.

Write the question in <question>...</question>.
For the answer, split the answer into the process towards reaching the answer and the final answer.
Write the process in <thoughts>thinking process of the writer</thoughts> and the final answer in <answer>...</answer>.

Directly copy, do not change the text.

Here is an example:

<text>
# Parallel plate capacitor: Proton vs Electron 1. Feb 28, 2017 ### Kmol6 1. The problem statement, all variables and given/known data A proton is released from rest at the positive plate of a parallel-plate capacitor. It crosses the capacitor and reaches the negative plate with a speed of 51000 m/s . What will be the final speed of an electron released from rest at the negative plate? 2. Relevant equations 3. The attempt at a solution I'm struggling as to what equations to use, I've tried using KE, assuming that the KE in the system is the same for both the Proton and the Electron, but that didn't work. I've also tried to find the magnitude of the electric field and work back to a kinematic equation, I'm so stuck . I could really just use some guidance as to where to start. Last edited by a moderator: Feb 28, 2017 2. Feb 28, 2017 ### Staff: Mentor Hi Kmol6, Since you don't know anything about the dimensions of the capacitor you're not likely to have much luck finding the electric field. When you say that you tried using KE, what does that mean exactly? What equations did you use (Hence the importance of the Relevant equations section of the formatting template)? Can you show your attempt? 3. Feb 28, 2017 ### Kmol6 KEi + PEi = KEf+PEf 1/2mv^2 +mgh= 1/2mv^2 + mgh 1/2(9.11x10^-31kg)(51000)^2 + 0 = 1/2 (1.67X10^-27)(V)^2 +0 Vf=1284 m/s 4. Feb 28, 2017 ### Staff: Mentor You've swapped the roles of the proton and electron. It was the proton that went first and ended up with a speed of 51000 m/s. Since it's not the conditions of the same particle that you are comparing, the conservation of energy law is not where you should start. What you're looking for is the formula that gives the work done on a charge falling through a given potential difference, hence the energy imparted. You can then claim that since the charges on the electron and proton are identical, they must both gain the same amount of kinetic energy. Then you can equate the KE's of each. 5. Feb 28, 2017 ### Kmol6 1/2mv^2=qDeltaV? Then sub the answer for delta V into DeltaU=qDeltaV using q as 1.602X10^-19C and then plug Delta U into 1/2mv^2=DeltaU and solve for v^2 of the electron? (I think systematically, combining equations isn't easy for me) I got 2.2X10^6m/s ? 6. Feb 28, 2017 ### Staff: Mentor That's the idea. Your result looks good. Note that since qΔV is the same for both particles you can write: $\frac{1}{2} m_ev_e^2 = q ΔV = \frac{1}{2} m_pv_p^2$ $m_ev_e^2 = m_pv_p^2$ $v_e = \sqrt{\frac{m_p}{m_e}}v_p$ 7. Feb 28, 2017 ### Kmol6 Thank you!!!!!
</text>
And this is how you should split it:
<question>The problem statement, all variables and given/known data A proton is released from rest at the positive plate of a parallel-plate capacitor. It crosses the capacitor and reaches the negative plate with a speed of 51000 m/s . What will be the final speed of an electron released from rest at the negative plate?</question>
<thoughts>I'm struggling as to what equations to use, I've tried using KE, assuming that the KE in the system is the same for both the Proton and the Electron, but that didn't work. I've also tried to find the magnitude of the electric field and work back to a kinematic equation, I'm so stuck . I could really just use some guidance as to where to start. Last edited by a moderator: Feb 28, 2017 2. Feb 28, 2017 ### Staff: Mentor Hi Kmol6, Since you don't know anything about the dimensions of the capacitor you're not likely to have much luck finding the electric field. When you say that you tried using KE, what does that mean exactly? What equations did you use (Hence the importance of the Relevant equations section of the formatting template)? Can you show your attempt? 3. Feb 28, 2017 ### Kmol6 KEi + PEi = KEf+PEf 1/2mv^2 +mgh= 1/2mv^2 + mgh 1/2(9.11x10^-31kg)(51000)^2 + 0 = 1/2 (1.67X10^-27)(V)^2 +0 Vf=1284 m/s 4. Feb 28, 2017 ### Staff: Mentor You've swapped the roles of the proton and electron. It was the proton that went first and ended up with a speed of 51000 m/s. Since it's not the conditions of the same particle that you are comparing, the conservation of energy law is not where you should start. What you're looking for is the formula that gives the work done on a charge falling through a given potential difference, hence the energy imparted. You can then claim that since the charges on the electron and proton are identical, they must both gain the same amount of kinetic energy. Then you can equate the KE's of each. 5. Feb 28, 2017 ### Kmol6 1/2mv^2=qDeltaV? Then sub the answer for delta V into DeltaU=qDeltaV using q as 1.602X10^-19C and then plug Delta U into 1/2mv^2=DeltaU and solve for v^2 of the electron? (I think systematically, combining equations isn't easy for me) I got 2.2X10^6m/s ? 6. Feb 28, 2017 ### Staff: Mentor That's the idea. Your result looks good. Note that since qΔV is the same for both particles you can write: $\frac{1}{2} m_ev_e^2 = q ΔV = \frac{1}{2} m_pv_p^2$ $m_ev_e^2 = m_pv_p^2$ $v_e = \sqrt{\frac{m_p}{m_e}}v_p$ 7. Feb 28, 2017 ### Kmol6 Thank you!!!!!</thoughts>
<answer>2.2X10^6m/s</answer>

Now do it for this text:""",
    'qa2': """Your goal is to split the text into a question, thought and an answer.

Make sure that the question is in the text. 
Make sure that the answer and the process of the writer to get to the answer are in the text.
Paraphrase the answer so that the answer is cleaned up. Make sure that the answer has the process of finding the solution.
Like backtracking, or verifying the answer, or setting subgoals.
Here are the definitions of the words:
Backtracking: The process of finding the solution by going back and forth between the answer and the question.
Verification: The process of checking the answer to see if it is correct.
Subgoal setting: The process of setting smaller goals to reach the final answer.

Write the question in <question>...</question>.
For the answer, split the answer into the process towards reaching the answer and the final answer.
Write the process in <thoughts>thinking process of the author with backtracking etc.</thoughts> and the final answer in <answer>...</answer>.
Use first person pronouns like "I" and "me" to refer to the author.
So, the thoughts should be in the first person, and should look like the author is thinking out loud. Eg: "I think I should try this next."
Include the mistakes made by the author in the thoughts section. If the author makes a mistake, include the mistake in the thoughts section.
Use present tense in the thoughts section. The thoughts section should look like the author is thinking out loud.
This will come with the realization from the author that they made a mistake.

Now do it for this text:""",
    'qa3': """Your goal is to split the text into a question, thought and an answer.

The thought is the process of the author to get to the answer.
The answer is the final answer.
Make sure that the thoughts have backtracking, verification, and subgoal setting.

Here are the definitions of the words:
Backtracking: The process of finding the solution by going back and forth between the answer and the question.
Verification: The process of checking the answer to see if it is correct.
Subgoal setting: The process of setting smaller goals to reach the final answer.

Write the question in <question>...</question>.
For the answer, split the answer into the process towards reaching the answer and the final answer.
Write the process in <thoughts> thinking process of the author with backtracking etc. goes here.</thoughts> and the final answer in <answer>...</answer>.
Use first person pronouns like "I" and "me" to refer to the author.
So, the thoughts should be in the first person, and should look like the author is thinking out loud. Eg: "I think I should try this next."
Include the mistakes made by the author in the thoughts section. If the author makes a mistake, include the mistake in the thoughts section.
Use present tense in the thoughts section. The thoughts section should look like the author is thinking out loud.
This will come with the realization from the author that they made a mistake.
Use about 500 words for the thoughts section.

Now do it for this text:""",

    'qa_none': """Your goal is to split the text into a question, thought and an answer.
Make sure that the question, thoughts and answer are in the text. 
Paraphrase the answer so that the answer is cleaned up. Make sure that the answer has steps to find the solution.    
Write the question in <question>...</question>.
Write the process in <thoughts>steps to find the solution</thoughts> and the final answer in <answer>...</answer>.
Use about 500 words for the thoughts section.

Now do it for this text:""",
}

    if args.dataset_name == 'finemath':
        ds = datasets.load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split=args.split, num_proc=os.cpu_count()-2)
    elif args.dataset_name == 'openwebmath':
        ds = datasets.load_dataset('Asap7772/open-web-math-none-processed-v2', num_proc=os.cpu_count()-2, split=args.split)
    elif args.dataset_name == 'openwebmath_backtrack':
        ds = datasets.load_dataset('Asap7772/open-web-math-backtrack-processed-v2', num_proc=os.cpu_count()-2, split=args.split)
    elif args.dataset_name == 'openwebmath_none':
        ds = datasets.load_dataset('Asap7772/open-web-math-none-processed-v2', num_proc=os.cpu_count()-2, split=args.split)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
        
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))
    
    # filter examples where 'contain_problem' is no or 'contain_solution' is no
    # if args.dataset_name == 'openwebmath' or args.dataset_name == 'openwebmath_backtrack':
    #     ds = ds.filter(lambda x: x['contain_problem'] != 'no' and x['contain_solution'] != 'no')
    #     print(f"Number of examples after filtering: {len(ds)}")

    llm = LLM(
        model='Qwen/Qwen2.5-32B-Instruct',
        tokenizer_mode="auto",
        max_num_seqs=64,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    num_batches = math.ceil(len(ds) / args.save_every)
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, tokenizer, prompt_templates)
        sampling_params = SamplingParams(
            max_tokens=4096+1024,
            temperature=0,
        )

        responses = llm.generate(prompts, sampling_params=sampling_params)

        outputs_dict = {
            'raw_qa': [None] * len(curr_batch),
            'query': [None] * len(curr_batch),
            'completion': [None] * len(curr_batch)
        }
        
        for i, response in enumerate(responses):
            output = response.outputs[0].text.strip()
            query, completion = parse_output(output)
            outputs_dict['raw_qa'][i] = output
            outputs_dict['query'][i] = query
            outputs_dict['completion'][i] = completion
        
        curr_batch = curr_batch.add_column('raw_qa', outputs_dict['raw_qa'])
        curr_batch = curr_batch.add_column('query', outputs_dict['query'])
        curr_batch = curr_batch.add_column('completion', outputs_dict['completion'])

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'owm_nonev4_{suffix}'
            ds_so_far.push_to_hub(ds_out_name)
        except Exception as e:
            print(f'Error saving dataset: {e}')
            continue
    
    try:
        ds_so_far = datasets.concatenate_datasets(all_ds)
        if args.start >= 0 and args.end >= 0 and args.start < args.end:
            suffix = f'_{args.start}_{args.end}'
        else:
            suffix = ''
        ds_out_name = f'owm_nonev4_{suffix}'
        ds_so_far.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
