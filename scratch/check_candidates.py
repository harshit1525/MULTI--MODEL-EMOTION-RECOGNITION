import json

for c in ['A', 'B']:
    with open(f'results/webcam_rafdb_exp{c}.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'=== Candidate {c} ===')
    clips = d['user_webcam_clips']['clips']
    for name, v in clips.items():
        print(f"{name:16s} GT: {v['ground_truth']:8s} Maj: {v['majority_vote_label']:8s} Correct: {str(v['majority_vote_correct']):5s} Acc: {v['frame_accuracy_pct']:5.1f}%")
    pooled = d['user_webcam_clips']['pooled_summary']
    print(f"Majority accuracy: {pooled['majority_accuracy_pct']}% ({pooled['majority_correct_count']}/{pooled['num_clips']})")
    print(f"Macro-F1 on clips: {pooled['macro_f1']}")
    print(f"RAF-DB test acc: {d['rafdb_test']['accuracy_pct']}%, macro-f1: {d['rafdb_test']['macro_f1']}")
    print(f"MELD test acc: {d['meld_heldout']['accuracy_pct']}%, macro-f1: {d['meld_heldout']['macro_f1']}")
    print()
