import os
import re

import datasets

def extract_variables(markdown_text):
    """
    Extracts variable values from a markdown string.
    The markdown is expected to have sections where a header (starting with '##')
    is immediately followed (after optional blank lines) by a single line
    containing the variable value.
    
    Args:
        markdown_text (str): The markdown content as a string.
        
    Returns:
        dict: A dictionary where keys are header texts and values are the extracted variable lines.
    """
    # This regex works as follows:
    # - It finds a header line that starts with '##', capturing any text that follows.
    # - It then matches one or more newline characters (allowing for blank lines),
    #   and captures the first non-empty line that follows as the variable value.
    pattern = r"##\s*(.*?)\s*\n+(?!##)([^\n]+)"
    
    matches = re.findall(pattern, markdown_text)
    
    variables = {}
    for header, var in matches:
        # Strip any extra whitespace from the variable text
        variables[header] = var.strip()
    return variables

def map_fn_backtrack(examples):
    ret_dict = {}
    # Process each sample in the batch
    for i in range(len(examples['text'])):
        curr_backtrack = examples['backtracking_raw'][i]
        try:
            extracted_vars = extract_variables(curr_backtrack)
        except Exception:
            extracted_vars = {}  # Use an empty dict if extraction fails
        
        required_keys = ['Thoughts', 'Does Backtrack?', 'Number of backtrack steps']
        mapped_keys = ['thoughts_backtrack', 'is_backtrack', 'backtrack_count']
        
        # Check if all required keys are present
        if all(key in extracted_vars for key in required_keys):
            for key, mapped_key in zip(required_keys, mapped_keys):
                ret_dict.setdefault(mapped_key, []).append(extracted_vars[key])
        else:
            # Append default values if extraction is incomplete
            for mapped_key in mapped_keys:
                ret_dict.setdefault(mapped_key, []).append(None)
        
        # Always append the original example values
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_backchain(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):
        curr_backchain = examples['backward_chaining_raw'][i]
        try:
            extracted_vars = extract_variables(curr_backchain)
        except Exception:
            extracted_vars = {}
        
        required_keys = ['Thoughts', 'Does the text exhibit backward chaining?', 'Number of backward chaining instances']
        mapped_keys = ['thoughts_backchain', 'is_backchain', 'backchain_count']
        
        if all(key in extracted_vars for key in required_keys):
            for key, mapped_key in zip(required_keys, mapped_keys):
                ret_dict.setdefault(mapped_key, []).append(extracted_vars[key])
        else:
            for mapped_key in mapped_keys:
                ret_dict.setdefault(mapped_key, []).append(None)
        
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_verification(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):            
        curr_verification = examples['verification_raw'][i]
        try:
            extracted_vars = extract_variables(curr_verification)
        except Exception:
            extracted_vars = {}
        
        required_keys = ['Thoughts', 'Does verification?', 'Number of answer verification steps']
        mapped_keys = ['thoughts_verification', 'is_verification', 'verification_count']
        
        if all(key in extracted_vars for key in required_keys):
            for key, mapped_key in zip(required_keys, mapped_keys):
                ret_dict.setdefault(mapped_key, []).append(extracted_vars[key])
        else:
            for mapped_key in mapped_keys:
                ret_dict.setdefault(mapped_key, []).append(None)
        
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_subgoal(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):
        curr_subgoal = examples['subgoal_setting_raw'][i]
        try:
            extracted_vars = extract_variables(curr_subgoal)
        except Exception:
            extracted_vars = {}
        
        required_keys = ['Thoughts', 'Does subgoal setting?', 'Number of subgoal setting steps']
        mapped_keys = ['thoughts_subgoal', 'is_subgoal', 'subgoal_count']
        
        if all(key in extracted_vars for key in required_keys):
            for key, mapped_key in zip(required_keys, mapped_keys):
                ret_dict.setdefault(mapped_key, []).append(extracted_vars[key])
        else:
            for mapped_key in mapped_keys:
                ret_dict.setdefault(mapped_key, []).append(None)
        
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

dataset_name = 'obiwan96/open_web_math_raw_v3_0_200000'
ds = datasets.load_dataset(dataset_name, split='train')

# Apply the mapping functions
ds = ds.map(map_fn_backtrack, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_backchain, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_verification, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_subgoal, batched=True, remove_columns=ds.column_names, num_proc=64)

ds.push_to_hub('obiwan96/open_web_math_raw_v3_0_200000_processed')



