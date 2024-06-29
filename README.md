# Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models

This repository contains the resource for our paper [Investigating and Mitigating the Multimodal Hallucination Snowballing in Large Vision-Language Models (ACL 2024)]()

Though advanced in understanding visual information with human languages, Large Vision-Language Models (LVLMs) still suffer from multimodal hallucinations. A natural concern is that during multimodal interaction, the generated hallucinations could influence the LVLMs' subsequent generation. Thus, we raise a question: *When presented with a query relevant to the previously generated hallucination, will LVLMs be misled and respond incorrectly, even though the ground visual information exists?* To answer this, we propose a framework called *MMHalSnowball* to evaluate LVLMs' behaviors when encountering generated hallucinations, where LVLMs are required to answer specific visual questions within a curated hallucinatory conversation. Crucially, our experiment shows that the performance of open-source LVLMs drops by at least 31%, indicating that LVLMs are prone to accept the generated hallucinations and make false claims that they would not have supported without distractions. We term this phenomenon *Multimodal Hallucination Snowballing*. To mitigate this, we further propose a training-free method called *Residual Visual Decoding*, where we revise the output distribution of LVLMs with the one derived from the residual visual input, providing models with direct access to the visual information. Experiments show that our method can mitigate more than 24% of the snowballed multimodal hallucination while maintaining capabilities.

## 1 Data Preparation
1. Clone this repo.
```shell
git clone https://github.com/whongzhong/MMHalSnowball.git
cd ./MMHalSnowball
```
2. Download the raw images from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)
```shell
cd ./evaluation
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
        "imageId": "2380767", // image id from the GQA dataset
        "answer": "no", // The answer that is consistent with the image
        "sample_id": "1016685", // sample id from the GQA dataset
        "hallucination_type": "imagination",
        "fact": "There is no tree in the image.", 
        "hallucinatory_fact": "There is a tree in the image.", // The modified fact sentence that is inconsistent with the image
        "modified_answer": "yes", // The hallucinatory answer that is consistent with the modified fact sentence
        "image_description": "In the image, a man is playing with a yellow frisbee under a bright blue sky. He has short hair, brown eyebrows, and blue eyes. The man is wearing a red shirt with white writing on it. His face shows a windblown cheek and closed lips. The frisbee is seen in the air, and the man is looking at it attentively. The bright clothing, specifically the red shirt, stands out in the photo, adding vibrancy to the scene.", // The image description that is consistent with the fact sentence
        "modified_description": "In the image, a man is playing with a yellow frisbee under a bright blue sky. He is wearing a red shirt with white writing on it. The man has short hair and blue eyes, with a windblown cheek and a closed mouth. In the background, there is a tree. The scene captures the joy of the man as he skillfully throws and catches the frisbee, enjoying the outdoors in the serene surroundings." // The hallucinatory description that is consistent with the modified fact sentence
    },
```
The structure of one sample in `utterance_{evaluation_task}_{conversation_setting}_{prompt_setting}.json` is as follows:
```shell
"1016685": { // sample id of the conversation, the annotation can be found in the corresponding annotation file
        "answer": "no", // The answer that is consistent with the image
        "modified_answer": "yes", // The hallucinatory answer that is consistent with the modified fact sentence
        "context_list": [ // the conversation strucure 
            {
                "role": "user",
                "type": "image",
                "content": "2380767" // image id of the input image
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
                "content": "Is there a tree in this image? Please answer the question using a single word or phrase." // The final evaluated user question
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
        "sample_id": "1016685",  // the sample id
        "original_answer": "no", // Adopted value of key 'answer' from utterance file
        "modified_answer": "yes", // Adopted value of key 'modified_answer' from utterance file
        "generated_answer": "Yes", // model generated response
    },
```
The model response should be put under the `evaluation/generation_results/{model_name}` folder:
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
        ├── generated_file_utterance_nocontextword_choice.json
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

