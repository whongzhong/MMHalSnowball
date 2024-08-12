import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
abspath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, abspath)

from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates, SeparatorStyle
from .llava.model.builder import load_pretrained_model
from .llava.utils import disable_torch_init
from .llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, abspath)

from ..rvd_sample import rvd_sampling
rvd_sampling()

from PIL import Image
import math
import random
random.seed(42)



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def context_generation(user_label, agent_label, conversation_input_dcit, mode="description"):
    #if not isinstance(conversation_input, list):
    #    conversation_input = [conversation_input]
    conversation_list = []
    if "description" in mode:
        conversation_input = conversation_input_dcit[mode]
        conversation_list.append((user_label, "Please describe the given image in detail"))
        conversation_list.append((agent_label, conversation_input))
    return conversation_list
# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, samples, image_folder, tokenizer, image_processor, model_config, conversation_dict_path, answer_prompt=None, shuffle=False):
        self.questions = samples
        if shuffle:
            random.shuffle(self.questions)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.shuffle = shuffle
        if answer_prompt is not None:
            self.answer_prompt = answer_prompt
        else:
            self.answer_prompt = "Answer the question using a single word or phrase."
            
        with open(conversation_dict_path, 'r') as f:
            self.conversation_dict = json.load(f)
            
        conv = conv_templates[args.conv_mode].copy()
        self.label_dict = {'user': conv.roles[0], 'agent': conv.roles[1]}
        

    def convert_conversation(self, conversation_list):
        
        converted_conversation_list = []
        image_tensor = None
        for single_utterance in conversation_list:
            if single_utterance['type'] == 'text':
                converted_conversation_list.append(self.construct_single_line(single_utterance))
            elif single_utterance['type'] == "image":
                image_tensor = self.construct_single_line(single_utterance)

            
        return image_tensor, converted_conversation_list

    def construct_single_line(self, message):
        # return with (role, message)
        if message['type'] == 'text':
            return [self.label_dict[message['role']], message['content']]
        
        # return processed image only
        elif message['type'] == "image":
            image = Image.open(os.path.join(self.image_folder, message['content'] + ".jpg")).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        return image_tensor

    def __getitem__(self, index):
        line = self.questions[index]
        
        context_dict = self.conversation_dict[line['sample_id']]
        
        conversation = context_dict['context_list']
        answer = context_dict['answer']
        modified_answer = context_dict['modified_answer']

        conv = conv_templates[args.conv_mode].copy()
        
        image_tensor, conversation_list = self.convert_conversation(conversation)
        
        # prepare raw input for Residual Visual Input and Blind Input (for Adaptive Distribution Blending)
        # Copy the conversation with the query only, omitting the dialog history
        rvd_list = [conversation_list[-1].copy()]
        adb_list = [conversation_list[-1].copy()]
        
        # for Residual Visual Input, prepend the visual information, the same as the original input
        # for blind input prepared for Adaptive Distribution Blending, we do not provide image information
        if self.model_config.mm_use_im_start_end:
            rvd_list[0][1] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + rvd_list[0][1]
            conversation_list[0][1] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + conversation_list[0][1]
        else:
            rvd_list[0][1] = DEFAULT_IMAGE_TOKEN + '\n' + rvd_list[0][1]
            conversation_list[0][1] = DEFAULT_IMAGE_TOKEN + '\n' + conversation_list[0][1]
            
        # initializing
        rvd_conv = conv_templates[args.conv_mode].copy()
        adb_conv = conv_templates[args.conv_mode].copy()
        
        # converting the input format
        for context in rvd_list:
            rvd_conv.append_message(context[0], context[1])
            
        rvd_conv.append_message(rvd_conv.roles[1], None)
        rvd_prompt = rvd_conv.get_prompt()
        
        for context in adb_list:
            adb_conv.append_message(context[0], context[1])
        adb_conv.append_message(adb_conv.roles[1], None)
        adb_prompt = adb_conv.get_prompt()
            
        for context in conversation_list:
            conv.append_message(context[0], context[1])
    
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        
        # generating input ids
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        rvd_input_ids = tokenizer_image_token(rvd_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        adb_input_ids = tokenizer_image_token(adb_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        return {'input_ids': input_ids, 'image_tensor': image_tensor, 'answer': answer, 'modified_answer': modified_answer, 'rvd_input_ids': rvd_input_ids, 'adb_input_ids': adb_input_ids}

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conversation_dict_path, batch_size=1, num_workers=0, shuffle_flag=False):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conversation_dict_path=conversation_dict_path, shuffle=shuffle_flag)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), 'r') as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, conversation_dict_path=args.conversation_file, shuffle_flag=True if args.canary_test > 0 else False)
    answer_list = []
    counting = 0
    
    for sample, line in tqdm(zip(data_loader, questions), total=len(questions)):
        if args.canary_test > 0 and counting > args.canary_test:
            break
        else:
            counting += 1
        idx = line["sample_id"]
        cur_prompt = line["question"]

        input_ids = sample['input_ids'].to(device='cuda', non_blocking=True)
        rvd_input_ids = sample['rvd_input_ids'].to(device='cuda', non_blocking=True) if args.rvd else None
        adb_input_ids = sample['adb_input_ids'].to(device='cuda', non_blocking=True) if args.adb else None
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                rvd_input_ids = rvd_input_ids,
                adb_input_ids = adb_input_ids,
                adb = args.adb,
                rvd = args.rvd,
                rvd_alpha = args.rvd_alpha,
                rvd_beta = args.rvd_beta,
                images=sample['image_tensor'].to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        answer_list.append({"sample_id": idx,
                                   "question": cur_prompt,
                                   "answer": line['answer'],
                                   "original_answer": sample["answer"][0],
                                   "imageId": line["imageId"],
                                   "fact": line["fact"],
                                   "hallucinatory_fact": line["hallucinatory_fact"] if "hallucinatory_fact" in line else "none",
                                   "modified_answer": sample["modified_answer"][0],
                                   "modified_description": line["modified_description"]  if "modified_description" in line else "none",
                                   "generated_answer": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}})
        # ans_file.flush()
    json.dump(answer_list, ans_file)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--vision-model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--conversation-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument('--rvd', action='store_true')
    parser.add_argument('--adb', action='store_true')
    parser.add_argument('--rvd-alpha', type=float, default=0)
    parser.add_argument('--rvd-beta', type=float, default=2.0)
    parser.add_argument('--canary_test', type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
