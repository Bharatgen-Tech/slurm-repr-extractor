import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

def check_file(item):
    line_num, line = item
    try:
        sample = json.loads(line)
        source = sample.get('source', '')
        exists = os.path.exists(source)
        return (exists, source)
    except:
        return (False, '')

#jsonl_path = "/fsx/kartik.jagtap/SLAM_ASR/Representations/Ashutosh_Work_DONOT_DELETE/mucs_fleurs_sar_original_file_paths.jsonl"
jsonl_path = "/fsx/ashutosh.adhikari/representations/data/as/input.jsonl"

# Load lines
with open(jsonl_path, 'r') as f:
    lines = [(i+1, line.strip()) for i, line in enumerate(f)]

print(f"Total lines: {len(lines)}\n")

exist = 0
missing = 0
missing_paths = []
start = time.time()

with ProcessPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(check_file, item) for item in lines]
    
    for future in tqdm(as_completed(futures), total=len(lines), desc="Processes"):
        exists, source = future.result()
        if exists:
            exist += 1
        else:
            missing += 1
            missing_paths.append(source)

elapsed = time.time() - start

# Dump missing paths
with open('missing2.txt', 'w') as f:
    for path in missing_paths:
        f.write(path + '\n')

print(f"\n{'='*60}")
print(f"EXIST:    {exist}")
print(f"MISSING:  {missing}")
print(f"TOTAL:    {len(lines)}")
print(f"PCT:      {(exist/len(lines)*100):.2f}%")
print(f"TIME:     {elapsed:.2f}s")
print(f"SPEED:    {len(lines)/elapsed:.0f} files/sec")
print(f"{'='*60}")
print(f"\nMissing paths saved to: missing.txt")

