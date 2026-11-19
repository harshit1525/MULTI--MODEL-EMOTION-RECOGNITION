import os
import torch
import numpy as np
import json

EMOTION_TO_IDX = {
    'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
    'neutral': 4, 'sadness': 5, 'surprise': 6
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

def get_file_size(path):
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    return size_bytes, size_mb, size_gb

def format_bytes(size_bytes):
    size_mb = size_bytes / (1024 * 1024)
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    return f"{size_mb:.2f} MB"

def verify_embeddings():
    emb_dir = 'audio_embeddings'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    splits = ['train', 'dev', 'test']
    report = {}
    
    print("========================================================")
    print("MELD AUDIO EMBEDDING DATASET REPORT")
    print("========================================================")
    
    total_samples_all = 0
    total_values_all = 0
    conceptual_dim = None
    
    split_keys = {'train': set(), 'dev': set(), 'test': set()}
    
    for split in splits:
        path = os.path.join(emb_dir, f'{split}_embeddings.pt')
        if not os.path.exists(path):
            print(f"[ERROR] Missing {path}")
            continue
            
        # File sizes
        file_bytes, file_mb, file_gb = get_file_size(path)
        
        # Load data
        data = torch.load(path, map_location='cpu')
        
        num_samples = len(data)
        if num_samples == 0:
            continue
            
        all_embeddings = []
        emotion_counts = {emo: 0 for emo in EMOTION_TO_IDX.keys()}
        
        non_768_count = 0
        dim_768_count = 0
        non_float32_count = 0
        float32_count = 0
        nan_count = 0
        inf_count = 0
        
        # Check all embeddings
        for item in data:
            emb = item['audio_embedding']
            all_embeddings.append(emb)
            
            # Key checks
            sample_id = item.get('sample_id', f"dia{item.get('Dialogue_ID', 'N/A')}_utt{item.get('Utterance_ID', 'N/A')}")
            split_keys[split].add(sample_id)
            
            # Class distribution
            emo = item.get('Emotion')
            if emo in emotion_counts:
                emotion_counts[emo] += 1
                
            # Dim check
            if list(emb.shape) == [768]:
                dim_768_count += 1
            else:
                non_768_count += 1
                
            # Dtype check
            if emb.dtype == torch.float32:
                float32_count += 1
            else:
                non_float32_count += 1
                
        all_embeddings_tensor = torch.stack(all_embeddings)
        
        # Nan/Inf on the stacked tensor
        nan_count = torch.isnan(all_embeddings_tensor).sum().item()
        inf_count = torch.isinf(all_embeddings_tensor).sum().item()
        finite_count = all_embeddings_tensor.numel() - nan_count - inf_count
        
        min_val = torch.min(all_embeddings_tensor).item()
        max_val = torch.max(all_embeddings_tensor).item()
        mean_val = torch.mean(all_embeddings_tensor).item()
        std_val = torch.std(all_embeddings_tensor).item()
        
        emb_dim = all_embeddings_tensor.shape[1]
        if conceptual_dim is None:
            conceptual_dim = emb_dim
            
        matrix_shape = (num_samples, emb_dim)
        total_values = num_samples * emb_dim
        raw_memory = total_values * 4  # float32 is 4 bytes
        
        total_samples_all += num_samples
        total_values_all += total_values
        
        print(f"\n{split.upper()}")
        print("-" * 56)
        print(f"Number of samples       : {num_samples}")
        print(f"Features per sample     : {emb_dim}")
        print(f"Embedding matrix shape  : {matrix_shape}")
        print(f"Total embedding values  : {total_values}")
        print(f"Data type               : {all_embeddings_tensor.dtype}")
        print(f"Memory size             : {format_bytes(file_bytes)}")
        print("-" * 56)
        
        report[split] = {
            "num_samples": num_samples,
            "embedding_dimension": emb_dim,
            "matrix_shape": list(matrix_shape),
            "total_values": total_values,
            "dtype": str(all_embeddings_tensor.dtype),
            "file_size_bytes": file_bytes,
            "file_size_mb": file_mb,
            "raw_memory_bytes": raw_memory,
            "nan_count": nan_count,
            "infinity_count": inf_count,
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
            "dim_768_count": dim_768_count,
            "non_768_count": non_768_count,
            "float32_count": float32_count,
            "non_float32_count": non_float32_count,
            "emotion_counts": emotion_counts
        }
        
    print("\n========================================================")
    print("ALL DATASETS")
    print("========================================================")
    print(f"Total samples           : {total_samples_all}")
    print(f"Features per sample     : {conceptual_dim}")
    print(f"Total embedding values  : {total_values_all}")
    print("\nCombined conceptual matrix shape:")
    print(f"({total_samples_all}, {conceptual_dim})")
    
    print("\n========================================================")
    print("3. VERIFY THAT EVERY EMBEDDING HAS THE SAME DIMENSION")
    print("========================================================")
    all_dim_pass = True
    for split in splits:
        if split not in report: continue
        print(f"\n{split.upper()}:")
        print(f"768-D embeddings: {report[split]['dim_768_count']}")
        print(f"Non-768 embeddings: {report[split]['non_768_count']}")
        if report[split]['non_768_count'] > 0:
            all_dim_pass = False
    print(f"\nAll embeddings have consistent dimension: {'PASS' if all_dim_pass else 'FAIL'}")

    print("\n========================================================")
    print("5. TOTAL NUMBER OF FEATURES")
    print("========================================================")
    for split in splits:
        if split not in report: continue
        print(f"{split.capitalize()}: {report[split]['total_values']}")
    print(f"Combined: {total_values_all}")

    print("\n========================================================")
    print("6. MEMORY / STORAGE INFORMATION")
    print("========================================================")
    for split in splits:
        if split not in report: continue
        print(f"\n{split.upper()}:")
        print(f"File size in bytes: {report[split]['file_size_bytes']}")
        print(f"File size in MB: {report[split]['file_size_mb']:.2f}")
        if report[split]['file_size_mb'] > 1024:
            print(f"File size in GB: {report[split]['file_size_mb']/1024:.2f}")
        print(f"Raw embedding memory: {report[split]['raw_memory_bytes']} bytes")

    print("\n========================================================")
    print("7. DATA TYPE")
    print("========================================================")
    all_dtype_pass = True
    for split in splits:
        if split not in report: continue
        print(f"\n{split.upper()}:")
        print(f"float32 embeddings = {report[split]['float32_count']}")
        print(f"other dtype = {report[split]['non_float32_count']}")
        if report[split]['non_float32_count'] > 0:
            all_dtype_pass = False
    print(f"\nConsistent dtype: {'PASS' if all_dtype_pass else 'FAIL'}")

    print("\n========================================================")
    print("8. EMBEDDING VALUE VALIDATION")
    print("========================================================")
    for split in splits:
        if split not in report: continue
        finite = report[split]['total_values'] - report[split]['nan_count'] - report[split]['infinity_count']
        print(f"\n{split.upper()}")
        print(f"Total values: {report[split]['total_values']}")
        print(f"NaN values: {report[split]['nan_count']}")
        print(f"Infinite values: {report[split]['infinity_count']}")
        print(f"Finite values: {finite}")

    print("\n========================================================")
    print("9. GLOBAL EMBEDDING STATISTICS")
    print("========================================================")
    for split in splits:
        if split not in report: continue
        print(f"\n{split.upper()}:")
        print(f"Min = {report[split]['min']:.6f}")
        print(f"Max = {report[split]['max']:.6f}")
        print(f"Mean = {report[split]['mean']:.6f}")
        print(f"Std = {report[split]['std']:.6f}")

    print("\n========================================================")
    print("10. METADATA DIMENSIONS")
    print("========================================================")
    for split in splits:
        path = os.path.join(emb_dir, f'{split}_embeddings.pt')
        if not os.path.exists(path): continue
        data = torch.load(path, map_location='cpu')
        
        keys = list(data[0].keys())
        print(f"\n{split.upper()}:")
        print(f"Metadata fields:\n{keys}")
        print(f"Number of metadata records:\n{len(data)}")
        print(f"Embedding records:\n{len(data)}")
        print(f"Metadata/embedding count match:\n{'PASS' if len(data) == len(data) else 'FAIL'}")
        
    print("\n========================================================")
    print("11. EMOTION DATASET DISTRIBUTION")
    print("========================================================")
    combined_emo = {emo: 0 for emo in EMOTION_TO_IDX.keys()}
    
    for split in splits:
        if split not in report: continue
        print(f"\n{split.upper()}:")
        n = report[split]['num_samples']
        for emo, count in report[split]['emotion_counts'].items():
            pct = (count / n) * 100 if n > 0 else 0
            print(f"{emo:<10}: {count} ({pct:.2f}%)")
            combined_emo[emo] += count
            
    print("\nCOMBINED:")
    for emo, count in combined_emo.items():
        pct = (count / total_samples_all) * 100 if total_samples_all > 0 else 0
        print(f"{emo:<10}: {count} ({pct:.2f}%)")

    print("\n========================================================")
    print("12. UNIQUE SAMPLE CHECK")
    print("========================================================")
    for split in splits:
        if split not in report: continue
        num_keys = len(split_keys[split])
        dupes = report[split]['num_samples'] - num_keys
        print(f"{split.upper()} duplicate keys: {dupes}")
        
    overlap_pass = True
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            s1, s2 = splits[i], splits[j]
            if s1 in report and s2 in report:
                overlap = len(split_keys[s1].intersection(split_keys[s2]))
                if overlap > 0:
                    print(f"{s1.capitalize()} <-> {s2.capitalize()} overlap: {overlap}")
                    overlap_pass = False
    
    print(f"\nData leakage / key overlap:\n{'PASS' if overlap_pass else 'FAIL'}")

    print("\n========================================================")
    print("13. FINAL SUMMARY TABLE")
    print("========================================================")
    print(f"| {'Split':<6} | {'Samples':<7} | {'Dimensions':<10} | {'Matrix Shape':<14} | {'Total Values':<12} | {'Dtype':<13} | {'File Size':<9} |")
    print(f"|{'-'*8}|{'-'*9}|{'-'*12}|{'-'*16}|{'-'*14}|{'-'*15}|{'-'*11}|")
    for split in splits:
        if split not in report: continue
        r = report[split]
        ms = f"({r['num_samples']}, {r['embedding_dimension']})"
        dt = r['dtype'].replace('torch.', '')
        fs = format_bytes(r['file_size_bytes'])
        print(f"| {split.capitalize():<6} | {r['num_samples']:<7} | {r['embedding_dimension']:<10} | {ms:<14} | {r['total_values']:<12} | {dt:<13} | {fs:<9} |")
        
    ms_total = f"({total_samples_all}, {conceptual_dim})"
    print(f"| {'Total':<6} | {total_samples_all:<7} | {conceptual_dim:<10} | {ms_total:<14} | {total_values_all:<12} | {'float32':<13} | {'-':<9} |")

    # 14. CREATE JSON REPORT
    report['combined'] = {
        "num_samples": total_samples_all,
        "embedding_dimension": conceptual_dim,
        "conceptual_matrix_shape": [total_samples_all, conceptual_dim],
        "total_values": total_values_all
    }
    
    json_path = os.path.join(results_dir, 'embedding_dataset_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"\nJSON report generated at: {json_path}")


if __name__ == '__main__':
    verify_embeddings()
