import datasets

dataset_name = 'obiwan96/open_web_math_raw_v3_0_200000_processed'
ds = datasets.load_dataset(dataset_name, split='train')

num_samples = len(ds)
# get counts of backtrack, backchain, verification, subgoal
backtrack_count = ds['backtrack_count']
backchain_count = ds['backchain_count']
verification_count = ds['verification_count']
subgoal_count = ds['subgoal_count']

# get number of samples with non-None counts
backtrack_count_none = sum(1 for count in backtrack_count if count is None)
backchain_count_none = sum(1 for count in backchain_count if count is None)
verification_count_none = sum(1 for count in verification_count if count is None)
subgoal_count_none = sum(1 for count in subgoal_count if count is None)

print(f'backtrack_count_none: {backtrack_count_none} ({backtrack_count_none / num_samples * 100}%)')
print(f'backchain_count_none: {backchain_count_none} ({backchain_count_none / num_samples * 100}%)')
print(f'verification_count_none: {verification_count_none} ({verification_count_none / num_samples * 100}%)')
print(f'subgoal_count_none: {subgoal_count_none} ({subgoal_count_none / num_samples * 100}%)')

# get number of samples with counts greater than 0
backtrack_non_none = [count for count in backtrack_count if count is not None]
backtrack_count_gt_0 = sum(1 for count in backtrack_non_none if count > 0)
backchain_non_none = [count for count in backchain_count if count is not None]
backchain_count_gt_0 = sum(1 for count in backchain_non_none if count > 0)
verification_non_none = [count for count in verification_count if count is not None]
verification_count_gt_0 = sum(1 for count in verification_non_none if count > 0)
subgoal_non_none = [count for count in subgoal_count if count is not None]
subgoal_count_gt_0 = sum(1 for count in subgoal_non_none if count > 0)

print(f'backtrack_count_gt_0: {backtrack_count_gt_0} ({backtrack_count_gt_0 / num_samples * 100}%)')
print(f'backchain_count_gt_0: {backchain_count_gt_0} ({backchain_count_gt_0 / num_samples * 100}%)')
print(f'verification_count_gt_0: {verification_count_gt_0} ({verification_count_gt_0 / num_samples * 100}%)')
print(f'subgoal_count_gt_0: {subgoal_count_gt_0} ({subgoal_count_gt_0 / num_samples * 100}%)')

# get average counts
backtrack_count_avg = sum(backtrack_count) / num_samples
backchain_count_avg = sum(backchain_count) / num_samples
verification_count_avg = sum(verification_count) / num_samples
subgoal_count_avg = sum(subgoal_count) / num_samples

print(f'backtrack_count_avg: {backtrack_count_avg}')
print(f'backchain_count_avg: {backchain_count_avg}')
print(f'verification_count_avg: {verification_count_avg}')
print(f'subgoal_count_avg: {subgoal_count_avg}')




