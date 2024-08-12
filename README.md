# Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models

This repository contains the resource for our paper [Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models (ACL 2024)](https://www.arxiv.org/abs/2407.00569)

Though advanced in understanding visual information with human languages, Large Vision-Language Models (LVLMs) still suffer from multimodal hallucinations. A natural concern is that during multimodal interaction, the generated hallucinations could influence the LVLMs' subsequent generation. Thus, we raise a question: *When presented with a query relevant to the previously generated hallucination, will LVLMs be misled and respond incorrectly, even though the ground visual information exists?* To answer this, we propose a framework called *MMHalSnowball* to evaluate LVLMs' behaviors when encountering generated hallucinations, where LVLMs are required to answer specific visual questions within a curated hallucinatory conversation. Crucially, our experiment shows that the performance of open-source LVLMs drops by at least 31%, indicating that LVLMs are prone to accept the generated hallucinations and make false claims that they would not have supported without distractions. We term this phenomenon *Multimodal Hallucination Snowballing*. To mitigate this, we further propose a training-free method called *Residual Visual Decoding*, where we revise the output distribution of LVLMs with the one derived from the residual visual input, providing models with direct access to the visual information. Experiments show that our method can mitigate more than 24% of the snowballed multimodal hallucination while maintaining capabilities.

## 1 Data Preparation
1. Clone this repo.
```shell
git clone https://github.com/whongzhong/MMHalSnowball.git
cd ./MMHalSnowball
```
2. Download the raw images from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)
```shell
cd ./evaluation/data
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip
cd ..
```
3. Data annotations and curated conversations are under the `evaluation/data` folder:
```
data
├── utterance # utterance_{evaluation_task}_{conversation_setting}_{prompt_setting}.json
│   ├── utterance_mmhalsnowball_cleanconv_formatting.json
│   ├── utterance_mmhalsnowball_cleanconv_question.json
│   ├── utterance_mmhalsnowball_factconv_formatting.json
│   ├── utterance_mmhalsnowball_halluconv_formatting.json
│   ├── utterance_mmhalsnowball_halluconv_question.json
│   ├── utterance_mmhalsnowball_irrconv_formatting.json
│   └── utterance_wpi_factconv_choice.json
├── mmhalsnowball_test.json # Annotation for MMHalSnowball
└── wpi_test.json # Annotation for the Who Provides This Image (WPI) task
```
Note that our curated conversation utterances and questions for each conversation setting are under the `evaluation/data/utterance` folder.
The annotation structure of one sample in `{task}_test.json` is as follows:
```shell
    {
        "question": "Is there a tree in this image?",
        "imageId": "2380767", # image id from the GQA dataset
        "answer": "no", # The answer that is consistent with the image
        "sample_id": "1016685", # sample id from the GQA dataset
        "hallucination_type": "imagination",
        "fact": "There is no tree in the image.", 
        "hallucinatory_fact": "There is a tree in the image.", # The modified fact sentence that is inconsistent with the image
        "modified_answer": "yes", # The hallucinatory answer that is consistent with the modified fact sentence
        "image_description": "In the image, a man is playing with a yellow frisbee under a bright blue sky. He has short hair, brown eyebrows, and blue eyes. The man is wearing a red shirt with white writing on it. His face shows a windblown cheek and closed lips. The frisbee is seen in the air, and the man is looking at it attentively. The bright clothing, specifically the red shirt, stands out in the photo, adding vibrancy to the scene.", # The image description that is consistent with the fact sentence
        "modified_description": "In the image, a man is playing with a yellow frisbee under a bright blue sky. He is wearing a red shirt with white writing on it. The man has short hair and blue eyes, with a windblown cheek and a closed mouth. In the background, there is a tree. The scene captures the joy of the man as he skillfully throws and catches the frisbee, enjoying the outdoors in the serene surroundings." # The hallucinatory description that is consistent with the modified fact sentence
    },
```
The structure of one sample in `utterance_{evaluation_task}_{conversation_setting}_{prompt_setting}.json` is as follows:
```shell
"1016685": { # sample id of the conversation, the annotation can be found in the corresponding annotation file
        "answer": "no", # The answer that is consistent with the image
        "modified_answer": "yes", # The hallucinatory answer that is consistent with the modified fact sentence
        "context_list": [ # the conversation strucure 
            {
                "role": "user",
                "type": "image",
                "content": "2380767" # image id of the input image
            },
            {
                "role": "user",
                "type": "text",
                "content": "Please describe the given image in detail."
            },
            {
                "role": "agent",
                "type": "text",
                "content": "In the image, a man is playing with a yellow frisbee under a bright blue sky. He is wearing a red shirt with white writing on it. The man has short hair and blue eyes, with a windblown cheek and a closed mouth. In the background, there is a tree. The scene captures the joy of the man as he skillfully throws and catches the frisbee, enjoying the outdoors in the serene surroundings."
            },
            {
                "role": "user",
                "type": "text",
                "content": "Is there a tree in this image? Please answer the question using a single word or phrase." # The final evaluated user question
            }
        ]
    },
```

## 2 Evaluation
### 2.1 Dataset Statistics
Our curated evaluation dataset contains 4,973 data samples. The detailed sample distribution is as follows:

<img width="629" alt="image" src="https://github.com/whongzhong/MMHalSnowball/assets/40679859/13e71b7a-dfd1-4647-ac66-fc1ccb19a2dc">

### 2.2 Generate Model Responses
To evaluate one LVLM with our MMhalsnowball, you can utilize the annotation file and the corresponding utterance as the input to generate model responses. The response should be a JSON file, where the structure for one single sample should contain the following keys and values:
```shell
    {
        "sample_id": "1016685",  # the sample id
        "original_answer": "no", # Adopted value of key 'answer' from utterance file
        "modified_answer": "yes", # Adopted value of key 'modified_answer' from utterance file
        "generated_answer": "Yes", # model generated response
    },
```
The model response can be put under the `evaluation/generation_results/{model_name}` folder:
```shell
generation_results
└── LLaVA1.5-7B # model_name
	├── results.txt # saved evaluation results
    ├── mmhalsnowball # generated responses for mmhalsnowball 
    │   ├── generated_file_utterance_mmhalsnowball_cleanconv_formatting.json
    │   ├── generated_file_utterance_mmhalsnowball_cleanconv_question.json
    │   ├── generated_file_utterance_mmhalsnowball_factconv_formatting.json
    │   ├── generated_file_utterance_mmhalsnowball_halluconv_formatting.json
    │   ├── generated_file_utterance_mmhalsnowball_halluconv_question.json
    │   └── generated_file_utterance_mmhalsnowball_irrelevant_formatting.json
    └── wpi # generated responses for WPI task 
        └── generated_file_utterance_wpi_factconv_choice.json
```
### 2.3 Evaluation
Once getting the generated responses, you can evaluate the model response with our evaluation code. Note that our evaluation is based on comparing results from two conversation settings, such as halluconv. and cleanconv. settings. 
An example shell script is in `evaluation/eval.sh`:
```shell
ROOT_PATH=$1 # root path of the generated model responses

PREFIX="generated_file_" # filename prefix of model responses 
TEST_MODEL_NAME="mmhalsnowball" # task type
KEY="original_answer" # key for the answer that is consistent with the given image
DICT_PATH="./evaluation/data/mmhalsnowball_test.json" # annotation file path

echo "**************************************"
echo "    evaluating for MMHalSnowball      "
echo "**************************************"
python -m evaluation.eval \
    --prefix $PREFIX \
    --file-path $ROOT_PATH/$TEST_MODEL_NAME \
    --dict-path $DICT_PATH \
    --key $KEY

echo ""
echo ""

TEST_MODEL_NAME="wpi"
KEY="original_answer"
DICT_PATH="./evaluation/data/wpi_test.json"

echo "**************************************"
echo "         evaluating for WPI           "
echo "**************************************"
python -m evaluation.eval \
    --prefix $PREFIX \
    --file-path $ROOT_PATH/$TEST_MODEL_NAME \
    --dict-path $DICT_PATH \
    --key $KEY \
    --wpi-task # evaluating the WPI task
```
To calculate accuracy for a single file, you can run this script:
```python
python -m evaluation.eval \
	--file-path file_path_to_the_generated_response \
	--dict-path file_path_to_annotation_file \
	--key "original_answer" \
	--eval-single-file \ # evaluate for a single file
	--eval-criteria "containing" \ # matching option or phrase, choosing from option and containing
	--single-filename file_path_to_the_target_file # filename for the single file to be evaluated
```
## 3 Residual Visual Decoding
### 3.1 Requirements
Please install the requirements following the specific LVLM. In the following Sections, we use [LLaVA](https://github.com/haotian-liu/LLaVA) as an example.
### 3.2 Integrating into LVLMs
We follow [VCD](https://github.com/DAMO-NLP-SG/VCD/tree/master) to integrate our *Residual Visual Decoding* into LVLMs. We illustrate the steps to modify the LVLMs:
First, replacing the original sampling function with our method by adding the following code to the main script:
```python
from residual_visual_decoding.rvd_sample import evolve_rvd_sampling
evolve_rvd_sampling()
```
Second, adding necessary parameters in the model `forward` function. For LLaVA, it's in `llava_llama.py`:
```python
adb_input_ids: Optional[torch.LongTensor] = None,
adb: Optional[bool] = None,
rvd_input_ids: Optional[torch.LongTensor] = None,
rvd: Optional[bool] = None,
rvd_alpha: Optional[Float] = None,
```
Third, updating the hyperparameter in the `generate` function:
```python
parser.add_argument('--rvd', action='store_true')
parser.add_argument('--blind-rvd', action='store_true')
parser.add_argument('--rvd-alpha', type=float, default=0)
parser.add_argument('--rvd-beta', type=float, default=2.0)
args = parser.parse_args()

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
```
Fourth, updating parameters for model's `forward()` function so that these added parameters can be input. For LLaVA, it's in `LlavaLlamaForCausalLM`:
```python
def forward(
	...
	rvd_input_ids = None,
	adb_input_ids = None,
    adb = None,
    rvd = None,
    rvd_alpha = None,
    rvd_beta = None,
    ...
) -> Union[Tuple, CausalLMOutputWithPast]:
```
fifth, customizing the `__getitem__` function in the dataset to provide *residual visual inputs* and *blind inputs* for *Adaptive Distribution Blending*. The corresponding inputs are $(v,x)$ and $(x)$, respectively, and the original input is $(v,h,x)$. Note that $v,h,x$ represents *visual input*, *dialog history*, and *current text query*.  
We use our MMhalsnowball evaluation dataset for LLaVA as an example. Note that we wrote functions to help *convert our conversations in the evaluation dataset* into the LLaVA format:
```python
def convert_conversation(self, conversation_list):

    converted_conversation_list = []
    image_tensor = None
    for single_utterance in conversation_list:
        if single_utterance['type'] == 'text': converted_conversation_list.append(self.construct_single_line(single_utterance))
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

    # Incorporating and assembling the conversation
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
    if self.model_config.mm_use_im_start_end:
        # for Residual Visual Input, prepend the visual information, the same as the original input
        # for blind input prepared for Adaptive Distribution Blending, we do not provide image information
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
```
### 3.3 Inference with RVD
You can run inference with the following script following examples in `residual_visual_decoding/LLaVA/llava_rvd_mmhalsnowball_inf.sh` and  `residual_visual_decoding/LLaVA/llava_rvd_wpi_inf.sh`
## 4 Citation
If you find our paper useful, please cite our paper:
```bibtex
@misc{zhong2024investigating,
      title={Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models}, 
      author={Weihong Zhong and Xiaocheng Feng and Liang Zhao and Qiming Li and Lei Huang and Yuxuan Gu and Weitao Ma and Yuan Xu and Bing Qin},
      year={2024},
      eprint={2407.00569},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.00569}, 
}
```