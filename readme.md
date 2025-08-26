# MedArk: Towards Proactive Asking with Imperfect Information in Medical Multi-turn Dialogues

## Model and Data Access
You can download our model (MedArk) on Huggingface:

| Model          | Backbone           | Access    |
| -------------- | ------------------ | ------------- |
| MedArk-1.5B  | Qwen2.5 1.5B  | [Link](https://huggingface.co/Bolin97/MedArk-1.5B) |
| MedArk-3B | Qwen2.5 3B | [Lnik](https://huggingface.co/Bolin97/MedArk-3b) |
| MedArk-7B | Qwen2.5 7B | [Lnik](https://huggingface.co/Bolin97/MedArk-7b) |

The local medical knowledge base for our model is provided on Huggingface, you can download it from [Lnik](https://huggingface.co/datasets/Bolin97/MedicalQA) 

## Introduction

Large language models (LLMs) cannot effectively collaborate with humans who provide imperfect information at the initial stage of the dialogue, unless they learn to proactively ask questions. Typically, the imperfect information is manifested in two aspects:

<div align="center">
  <img src="https://github.com/user-attachments/assets/82566937-ebd0-418d-a7ce-4f47a640fb32" style="width:45%; height:auto;" />
</div>


Our core idea is to enable LLMs to decide whether to take the action of "ask" or "tell" at each turn by self-reasoning, with the belief of the decisions enhanced by retrieving knowledge related to the user input. Thus, we propose the ask and retrieve knowledge framework (Ark), where LLMs think through what to retrieve, when to stop retrieving, and then take actions accordingly.

Online medical consultations provide an ideal setting for imperfect information scenarios, where patients’ initial information is often imperfect, requiring doctors to ask for more details to provide reliable answers. However, the action paths between patient inputs and doctor responses are lacking; thus, we use Ark framework to fill in the paths given the doctor’s final action at each turn of the dialogue.

Then, we train the model (MedArk) using the data produced by Ark, which equips LLMs with the ability to **proactively ask questions to fill information gaps during conversations with patients, actively retrieve knowledge to mitigate medical hallucinations, and actively reason to decide next actions**.

## Use cases

<table>
  <tr>
    <td align="left" width="50%" style="word-wrap: break-word;" >
      <b>Interaction between MedArk and a patient.</b><br/>
      The key sentences of self-reasoning are marked in red,
      the actions taken after self-reasoning are marked in green,
      and the final response is marked in blue. Entire action path= $\mathcal{R} \to \mathcal{R} \to \mathcal{A}$.
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/e7786250-51f8-4833-9df9-058416e17547" style="width:80%; height:auto;"/>
    </td>
  </tr>
</table>


## User Manual for the Code

This is a guide on how to

1. generate data from medical dialogues.
2. train your own MedArk using your own data.
3. deploy it.

The current version is a prototype that's only meant for demonstration purposes. So it's not yet ready for serious use, and the way to use is not user-friendly.

In the following text, we define the following type,

```python
class Conversation(TypedDict):
    patient: str
    doctor: str
```

To use this prototype, the following procedures are required:

### Creating tagged.pkl

Create a dictionary that takes the hash of the first conversation of each conversation, and map it to the whole conversation. Save it as `tagged.pkl` under the tag_conversation folder. For example, for this following conversation:

```python
test_conv = [
    {
        'patient':'脸上一直出豆豆。脸上出豆，用过很多东西，总是反反复复，都快没信心了。',
        'doctor':'需要本人过来看看，特别是首诊很重要，以便准确诊断和确定治疗方案。'
    },
    {
        'patient':'一定要去吗？主要太远了我现在就是想咨询一下我需要做些什么，吃哪些药好一点，',
        'doctor':'皮肤科的病需要直观面诊。'
    }
]
```

You should calculate the hash using,

```python
def cal_hash(s: str) -> str:
    hash_object = hashlib.sha256(s.encode())  # 必须将字符串编码为字节
    hash_hex = hash_object.hexdigest() 
    return hash_hex
h = cal_hash(str(test_conv[0]))
```

Adn add a new item to the dictionary,

```python
dict_to_save[h] = test_conv
```

Then pickle the `dict_to_save` as `tagged.pkl`.

You may use `tag-conversation/construct_tagged.ipynb` for reference.

### Prepare the VectorDatabase

Then, use `docs/create_chroma_db.ipynb` to create a chroma database. You should supply the documents by your own. Each document should be a question-answer pair, with the `page_content` being the question, and having `answer` and `question` as key in meta data, each referring to the supplied documents.

For example, for,

```python
test_doc = {
    'q': 'this is a question',
    'a': 'this is an answer',
}

vector_store.add_documents(
    [
        Document(
            page_content=test_doc['q'],
            metadata={
                "question": test_doc['q'],
                "answer": test_doc['a']
            }
        )
    ]
)
```

The db will be automatically persisted to the `docs/chroma` folder.

### Prepare SFT Data

Run `construction-sft/first_pass_gen*.py`, you may use one with or without the extra search engine. Fill in the corresponding parameters in the script, and run it. It will create a `data_with_reasonigns_0.pkl`. You should supply the data in the format of `list[list[Conversation]]`. 

Run `construction-sft/regen_entiti.ipynb`. It will replace `data_with_reasonigns_0.pkl` with a new one that has extra informaitons contained.

Run `construction-sft/add_consersation_id.ipynb`. It will replace `data_with_reasonigns_0.pkl` with a new one that has conversation id.

Run `construction-sft/second_pass_gen.ipynb`, it will create `rendered_prompts.json`. Please note that you need a golang environment configured.

Run `construction-sft/tokenize.ipynb`, it will create a `construction-sft/train` dataset. Use this dataset to do SFT training.

### Train by SFT

Train the model with the data constructed from the previous stage with `train-sft/sft-train.py`.

### Prepare DPO Rejected Data

Go to `construction-dpo`. Run the `frist_pass_gen*.py`, which is identical to the SFT construction. Then run `infer.py` to do inference using the SFT adapter, it will be used as the rejection data for DPO training, which outputs a `records.pkl`.

### Prepare DPO Reasoning

After that, run `generate_new_reasoning.py` to generate reasoning data for DPO training. It will output a json file.

### Generate DPO Data

Run `construction-dpo/genetate_dpo_data.ipynb` to generate the DPO training data, it will give a json file.

### Train by DPO

Train the model with the data constructed from the previous stage with `train-dpo/dpo-train.py`. It will modify the weights of the SFT addapter.

### Deploy the Model

Use `deploy/depl_web_ui.py` to deploy the model using Gradio.

### Infer with the Trained Model

Run `offline-inference/infer.py` to do inference with the trained model. It will output a `records.pkl` that contains the conversation data.

### Evaluation

Go to the evaluation folder, fill in the given parameters. The `input` folder accepts the conversation data from other models. The `input-pkl` accepts the `records.pkl` generated from the `infer.py`. The `mrag` accetps the json data generated from MedRAG system with tweaked prompts. You may run `eval.py` to evaluate `input` folder, `eval_own.py` to evaluate `input-pkl`, and `eval_mrag.py` to evaluate `mrag`. Each of them accepts a number that indicates a offset. For example, `python eval_won.py 1` will do evaluation on the 3rd and 4th pickle file under the `input-pkl` folder. `eval*-jac.py` is also for evaluation, but yields differents metrics, only for the metrics related to ask utilities.

We have only put in each folder a demonstration file with model responses. For all the model responses, check the `evaluation-data` folder and read its readme.

## Citation
```
@inproceedings{medask,
  title={Ask and Retrieve Knowledge: Towards Proactive Asking with Imperfect Information in Medical Multi-turn Dialogues},
  author={Zhang, Bolin and Wang, Shengwei and Jiang, Yangqin and Sui, Dianbo and Tu, Zhiying and Chu, Dianhui},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1055--1065},
  year={2025}
}
```
