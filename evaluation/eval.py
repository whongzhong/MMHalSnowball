import json 
import os
import argparse
import random
random.seed(42)

import sys
abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, abspath)

def evaluating_criteria(golden, pred, criterion, additional_option=None, **kwargs):
    if criterion == "em":
        golden = golden.lower()
        pred = pred.lower()
        if golden == pred:
            return True
        else:
            return False
    elif criterion == "containing":
        golden = golden.lower()
        pred = pred.lower()
        if golden in pred:
            if additional_option is not None and additional_option in pred:
                return False
            return True
        else:
            return False
    elif criterion == "option":
        if golden in pred:
            if additional_option is not None and additional_option in pred:
                return False
            return True
        else:
            return False
    elif criterion == "":
        pass
    

def eval_gqa(pred, gold, criterion, additional_options=None):
    assert len(pred) == len(gold)
    total_sample = len(pred)
    accurate = 0
    for pred_answer, gold_answer, additional_option in zip(pred, gold, additional_options):
        if evaluating_criteria(gold_answer, pred_answer, criterion, additional_option):
            accurate += 1
            
    return {'accuracy': accurate * 1.0 / total_sample} if total_sample > 0 else {'accuracy': 0.0}

def count_difference(file_path, input_file_name, original_item, check_item, difference_label, criterion, additional_option):
    with open(os.path.join(file_path, input_file_name), 'r') as f1, open(os.path.join(file_path, difference_label + "_" + input_file_name), 'w') as f2:
        samples = json.load(f1)
        for line in samples:
            sample = line
            if evaluating_criteria(sample[original_item], sample[check_item], criterion, additional_option=sample[additional_option]):
                if difference_label not in sample:
                    sample[difference_label] = "True"
            else:
                if difference_label not in sample:
                    sample[difference_label] = "False"
                    
        
        json.dump(samples, f2)

def mark_flip(file_path, a_file_name, b_file_name, prefix, tag=None):
    a_setting = a_file_name.replace(prefix, "")
    b_setting = b_file_name.replace(prefix, "")
    with open(os.path.join(file_path, a_file_name), 'r') as fa, \
        open(os.path.join(file_path, b_file_name), 'r') as fb, \
        open(os.path.join(file_path, "flipcheck_" + a_setting + "_to_" + b_setting), 'w') as foa, \
        open(os.path.join(file_path, "flipcheck_" + b_setting + "_to_" + a_setting), 'w') as fob:
            
        
        samples_a = json.load(fa)
        samples_b = json.load(fb)
            
        total_sample = 0
        flip_sample = 0
        weak_flip_sample = 0
        for sample_a, sample_b in zip(samples_a, samples_b):
            
            if "sample_id" in sample_a:
                sample_a_id = sample_a['sample_id']
            else:
                sample_a_id = sample_a['question_id']
                
            if "sample_id" in sample_b:
                sample_b_id = sample_b['sample_id']
            else:
                sample_b_id = sample_b['question_id']

            if tag is None or ((tag and sample_a_id in pos_tag_dict)and pos_tag_dict[sample_a_id]["hallucination_type"] == tag):
                if sample_a["correct"] == "True": # fliprate total count based on the correct answers
                    total_sample += 1
            if sample_a["correct"] == "True" and sample_b["cheated"] == "True":
                if tag is None or ((tag and sample_b_id in pos_tag_dict) and pos_tag_dict[sample_b_id]["hallucination_type"] == tag):
                    flip_sample += 1
                    weak_flip_sample += 1
                sample_a['flipped'] = "True"
                sample_b['flipped'] = "True"
            elif sample_a["correct"] == "True" and sample_b["correct"] == "False":
                if tag is None or ((tag and sample_a_id in pos_tag_dict) and pos_tag_dict[sample_a_id]["hallucination_type"] == tag):
                    weak_flip_sample += 1
                sample_a['flipped'] = "Weak"
                sample_b['flipped'] = "Weak"
            else:
                sample_a['flipped'] = "False"
                sample_b['flipped'] = "False"
                
        json.dump(samples_a, foa)
        json.dump(samples_b, fob)
    
    fr = (flip_sample * 1.0 / total_sample)  if total_sample > 0 else 0
    print(f"Flip rate: {fr}")
    wfr = (weak_flip_sample * 1.0 / total_sample)  if total_sample > 0 else 0
    print(f"Weak flip rate: {wfr}")
    return fr, wfr

def eval_generated_answer(file_path, input_file_name, criterion, tag=None, key="original_answer"):
    gold = []
    pred = []
    modified = []
    
    with open(os.path.join(file_path, input_file_name), 'r') as f:
        
        samples = json.load(f)
            
        for line in samples:
            sample = line
                
            if "sample_id" in sample:
                sample_id = sample['sample_id']
            else:
                sample_id = sample['question_id']
            if (tag and sample_id in pos_tag_dict) and pos_tag_dict[sample_id]["hallucination_type"] != tag:
                continue
            gold.append(sample[key])
            modified.append(sample['modified_answer'])
            pred.append(sample['generated_answer'])
    print(f"> evaluating file: {input_file_name}")
    print("accuracy: ")
    acc = eval_gqa(pred, gold, criterion, additional_options=modified if criterion in ["option", "containing"] else [None for i in range(len(gold))])
    print(acc)
    modified_acc = eval_gqa(pred, modified, criterion, additional_options=gold if criterion in ["option", "containing"] else [None for i in range(len(modified))])
    print("modified accuracy: ")
    print(modified_acc)
    count_difference(file_path, input_file_name, key, "generated_answer", "correct", criterion, additional_option="modified_answer")
    count_difference(file_path, "correct" + "_" + input_file_name, "modified_answer", "generated_answer", "cheated", criterion, additional_option=key)
    return acc['accuracy'], modified_acc['accuracy']
    
def counting_flip(file_path, a_file_name, b_file_name, criterion, prefix, tag_list, key):
    label = "cheated_correct_"
    for tag in tag_list:
        try:
            print(f"======================evaluation for {tag} hallucination======================")
            b_acc, b_modified_acc = eval_generated_answer(file_path, b_file_name, criterion, tag, key=key)
            a_acc, a_modified_acc = eval_generated_answer(file_path, a_file_name, criterion, tag, key=key)
            a_acc = float('{:.4f}'.format(a_acc))
            b_acc = float('{:.4f}'.format(b_acc))
            print(f"delta-acc: {a_acc-b_acc}")
            print(f"scaled delta-acc: {a_acc*(a_acc-b_acc)}")
            fr, wfr= mark_flip(file_path, label + a_file_name, label + b_file_name, label + prefix, tag)
            
            print(f"accuracy for file {a_file_name}: {a_acc}")
            print(f"accuracy for file {b_file_name}: {b_acc}")
            print(f"Flip Rate: {fr}")
            print(f"Weak Flip Rate: {wfr}")
            
        except FileNotFoundError as e:
            pass

    
    print(f"======================evaluation for the whole======================")
    b_acc, b_modified_acc = eval_generated_answer(file_path, b_file_name, criterion, key=key)
    a_acc, a_modified_acc = eval_generated_answer(file_path, a_file_name, criterion, key=key)
    a_acc = float('{:.4f}'.format(a_acc))
    b_acc = float('{:.4f}'.format(b_acc))
    
    fr, wfr = mark_flip(file_path, label + a_file_name, label + b_file_name, label + prefix)
    
    print(f"accuracy for file {a_file_name}: {a_acc}")
    print(f"accuracy for file {b_file_name}: {b_acc}")
    print(f"Flip Rate: {fr}")
    print(f"Weak Flip Rate: {wfr}")
    

pos_tag_dict = {}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="yourfileprefix")
    parser.add_argument("--file-path", type=str, default="yourfilepath")
    parser.add_argument("--dict-path", type=str, default="yourdictpath")
    parser.add_argument("--key", type=str, default="generated_answer_key")
    parser.add_argument("--wpi-task", action='store_true')
    args = parser.parse_args()
    
    if not args.wpi_task:
        with open(args.dict_path, 'r') as f:
            samples = json.load(f)
            for sample in samples:
                pos_tag_dict[sample['sample_id']] = sample
        
        
        conversation_prefix_list =[]
        
        conversation_prefix_list.append(["utterance_mmhalsnowball_halluconv_question", "utterance_mmhalsnowball_cleanconv_question", 'containing'])
        conversation_prefix_list.append(["utterance_mmhalsnowball_halluconv_formatting", "utterance_mmhalsnowball_cleanconv_formatting", 'containing'])
        
        conversation_prefix_list.append(["utterance_mmhalsnowball_factconv_formatting", "utterance_mmhalsnowball_cleanconv_formatting", 'containing'])
        conversation_prefix_list.append(["utterance_mmhalsnowball_irrelevant_formatting", "utterance_mmhalsnowball_cleanconv_formatting", 'containing'])
        
        tag_list = ["relation", "attribute", "existence", "imagination"]
        for conversation_prefix in conversation_prefix_list:
            try:
                print(f"====================================")
                print(f"evaluation for {conversation_prefix}")
                print(f"====================================")
                a_file_name = args.prefix + conversation_prefix[1] + ".json"
                b_file_name = args.prefix + conversation_prefix[0] + ".json"
                counting_flip(args.file_path, a_file_name, b_file_name, criterion=conversation_prefix[2], prefix=args.prefix, tag_list=tag_list, key=args.key)
            except FileNotFoundError as e:
                pass
    else:
        file_name = args.prefix + "utterance_wpi_factconv_choice" + ".json"
        a_acc, a_modified_acc = eval_generated_answer(args.file_path, file_name, "option", key=args.key)
        print(f"accuracy for file {file_name}: {a_acc}")
        