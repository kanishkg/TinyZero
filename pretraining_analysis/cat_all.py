import datasets
all_ds_names = [
    'Asap7772/Asap7772open_web_math_raw_933334_966667',
    'Asap7772/Asap7772open_web_math_raw_700001_733334',
    'Asap7772/Asap7772open_web_math_raw_766668_800001',
    'Asap7772/Asap7772open_web_math_raw_666668_700001',
    'Asap7772/Asap7772open_web_math_raw_533335_566668',
    'Asap7772/Asap7772open_web_math_raw_400002_433335',
    'Asap7772/Asap7772open_web_math_raw_366668_400002',
    'Asap7772/Asap7772open_web_math_raw_633335_666668',
    'Asap7772/Asap7772open_web_math_raw_300001_333334',
    'Asap7772/Asap7772open_web_math_raw_966667_1000000',
    'Asap7772/Asap7772open_web_math_raw_466668_500002',
    'Asap7772/Asap7772open_web_math_raw_900001_933334',
    'Asap7772/Asap7772open_web_math_raw_333334_366668',
    'Asap7772/Asap7772open_web_math_raw_866667_900001',
    'Asap7772/Asap7772open_web_math_raw_500002_533335',
    'Asap7772/Asap7772open_web_math_raw_266668_300001',
    'Asap7772/Asap7772open_web_math_raw_566668_600001',
    'Asap7772/Asap7772open_web_math_raw_200000_233334',
    'Asap7772/Asap7772open_web_math_raw_233334_266668',
    'Asap7772/Asap7772open_web_math_raw_800001_833334',
    'Asap7772/Asap7772open_web_math_raw_433335_466668',
    'Asap7772/Asap7772open_web_math_raw_833334_866667',
    'Asap7772/Asap7772open_web_math_raw_600001_633335',
    'Asap7772/Asap7772open_web_math_raw_733334_766668',
    'obiwan96/obiwan96open_web_math_raw_0_15000',
    'obiwan96/obiwan96open_web_math_raw_15000_30000',
    'obiwan96/obiwan96open_web_math_raw_70000_115000',
    'obiwan96/obiwan96open_web_math_raw_115000_160000',
    'obiwan96/obiwan96open_web_math_raw_160000_200000',
    'obiwan96/obiwan96open_web_math_raw_30000_70000',
]

all_ds = [datasets.load_dataset(ds_name, split='train') for ds_name in all_ds_names]
all_ds = datasets.concatenate_datasets(all_ds)
all_ds.push_to_hub('Asap7772/open_web_math_raw_0_1000000')