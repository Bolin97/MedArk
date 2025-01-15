openai_api_key = ''
# an indexer for getting the conversation from conversation id
conv_id_to_c = pickle.load(open("../tag_conversation/tagged.pkl", "rb"))
bert_model_path = ''

import json
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from datasets import load_from_disk
from langchain_community.retrievers import BM25Retriever
import pickle

# corpus = pickle.load(open("./dx_corpus.pkl", "rb"))
# corpus = pickle.load(open("./ki_corpus.pkl", "rb"))
retriver = retriver = BM25Retriever.from_texts(["abc"], k=1)


from typing import Callable

def with_retry(f: Callable):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(f"Retry failed with error: {e}, retrying...")
                continue
    return wrapper

class IndependentFacts(BaseModel):
    """The independent facts of the given message. Should contain no more content than what's related to the query."""
    facts: list[str] = Field(
        description="应当是中文。The independent facts of the given message. Should contain no more content than what's related to the query."
    )

@with_retry
def get_independent_facts(i: str) -> list[str]:
    prompt = """Please breakdown the following sentence into independent facts. You must give them out in the form of IndependentFacts.\n{}""".format(i)
    llm_get_facts = llm.with_structured_output(IndependentFacts)
    facts = llm_get_facts.invoke(prompt)
    return facts.facts

class IsSupported(BaseModel):
    is_supported: bool = Field(
        description="Whether the fact is supported or not",
    )

@with_retry
def check_independent_facts(i: str) -> bool:
    docs = retriver.invoke(i)
    prompt = """Fact: {}
Reference: 

{}

Please verify whether the Fact is factually correct based on the reference and answer Supported or Not-supported.
Supported: you found the Reference that indicates that the fact is definitely correct.
Not-supported: not "Supported", either because you found the Reference that indicates the fact is incorrect, or the fact is unverifable.

You should give out the answer in the form of IsSupported.
""".format(
        i,
        ["\n".join([
            d.page_content for d in docs
        ])],
    )
    llm_get_facts = llm.with_structured_output(IsSupported).invoke(prompt)
    return llm_get_facts.is_supported

def get_fact_res(i: str) -> tuple[int, int]:
    facts = get_independent_facts(i)
    cnt = 0
    for fact in facts:
        if check_independent_facts(fact):
            cnt += 1
    return len(facts), cnt


llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
    temperature=0.2
)
class IsAsking(BaseModel):
    is_asking: bool = Field(
        description="Whether the user is asking a question or not",
    )
def is_ask(i: str, cnt: int = 0) -> bool:
    if isinstance(i, dict):
        return i['is_asking']

    try:
        llm_s = llm.with_structured_output(IsAsking)
        r = llm_s.invoke("请判断下面的句子是不是问句，并以 IsAsking 形式给出。\n{}".format(i))
        if r is None and cnt < 3:
            return is_ask(i, cnt + 1)
        else:
            if r is None:
                return False
            return r.is_asking
    except:
        return is_ask(i, cnt + 1)


llm = ChatOpenAI(
    api_key="sk-sBOt7smiSOaUAtA9KQXLJx4spWvomQbNX8BgRTxOuvtKHzvn",
    model="gpt-4o-mini",
    base_url="https://api.key77qiqi.cn/v1"
)

class AskingPart(BaseModel):

    asking: str = Field(
        description="The part of the sentence that the teacher asked."
    )
prompt_ask = "重复句子\n\n{}\n\n的问句部分，以 AskingPart 的方式给出。"


prompt = """在医患对话场景中，医生通常会提问以下几种对象。

- 疾病：即具体的疾病名称。
- 临床表现：即患者的症状，例如：头疼。
- 药物：即具体的药品名称，例如：阿司匹林。
- 手术：即具体的手术名称，例如：阑尾切除，心脏搭桥。
- 身体部位：即身体部位，例如：头，颈，胸，背，腰，腿，足。
- 检查项目：包括检查项目，检查结果，检查结论。
- 问诊医学概念：包括性别,年龄, 职业, 发病时间, 伴随症状，饮食情况，症状程度，曾用药，是否手术等等。

现在，对于以下这个医生的提问，

{}

给出医生提问中的所有对象，以 MedicalAskContent 的格式给出。

注意，你给出的所有内容必须出现在医生的提问中。每种对象可能不存在，也可能存在任意数目个。
"""

class MedicalAskContent(BaseModel):
    
    disease: list[str] = Field(
        default_factory=list,
        description="医生提问中的疾病对象"
    )
    symptom: list[str] = Field(
        default_factory=list,
        description="医生提问中的症状对象"
    )
    medcine: list[str] = Field(
        default_factory=list,
        description="医生提问中的药物对象"
    )
    surgery: list[str] = Field(
        default_factory=list,
        description="医生提问中的手术对象"
    )
    body_part: list[str] = Field(
        default_factory=list,
        description="医生提问中的身体部位对象"
    )
    medical_check: list[str] = Field(
        default_factory=list,
        description="医生提问中的检查项目对象"
    )
    concept: list[str] = Field(
        default_factory=list,
        description="医生提问中的问诊医学概念对象"
    )

@with_retry
def generate_entities(i: str) -> MedicalAskContent:
    llm_get_ask = llm.with_structured_output(AskingPart)
    r = llm_get_ask.invoke(prompt_ask.format(i))
    llm_get_entities = llm.with_structured_output(MedicalAskContent)
    return llm_get_entities.invoke(prompt.format(r))

import os
to_test = os.listdir("./mrag")
import json
from tqdm import tqdm
import Levenshtein
import hashlib

def cal_hash(s: str) -> str:
    hash_object = hashlib.sha256(s.encode())  # 必须将字符串编码为字节
    hash_hex = hash_object.hexdigest() 
    return hash_hex

def flatten_medical_ask_content(i: MedicalAskContent) -> str:
    return f"{''.join(i.disease)}{''.join(i.symptom)} {''.join(i.medcine)}{''.join(i.surgery)}{''.join(i.body_part)}{''.join(i.medical_check)}{''.join(i.concept)}"

class Presence(BaseModel):
    """The state of presence of the given medical objects in the given message. Each given medical object should either be present or not present in the given message."""
    reasoning: str = Field(
        description="The reason why you assert that the given medical objects are present or not present in the given message, or why you cannot assert it. Reasoning should be concise and contain only important points."
    )
    presence: list[str] = Field(
        description="The medical objects that are present in the given message."
    )
    non_presence: list[str] = Field(
        description="The medical objects that are not present in the given message."
    )

@with_retry
def check_presence(everything: list[str], msg) -> tuple[int, int]:
    prompt = f"""You need to check whether the following object presents in the given message.

Please note that, for each given object, it either is present or not present in the given message.

Please return in the form of Presence.

The objects are,

{json.dumps(everything, ensure_ascii=False)}

The messages are,

{msg}
    
"""
    res = llm.with_structured_output(Presence).invoke(prompt)
    return len(res.presence), len(res.non_presence)

@with_retry
def get_auv(model_out, objs: MedicalAskContent):
    conversation = model_out['src']['conversation']
    conv_id = cal_hash(str(conversation[0]))
    conv = conv_id_to_c[conv_id]
    cur_round = len(model_out['src']['conversation'])
    stacked_patients = "\n".join([
        e['patient'] for e in conv[cur_round:]
    ])
    return check_presence(
        (
            objs.disease + objs.symptom + objs.medcine + objs.surgery + objs.body_part + objs.medical_check + objs.concept
        ),
        stacked_patients
    )
import loguru
logger = loguru.logger
logger.remove(0)
logger.add("logs.log", level="INFO")
tt = []
for each in to_test:
    if os.path.exists("./output/{}.txt".format(each)):
        continue
    else:
        tt.append(each)

@with_retry
def get_is_informational(q, model_out, ref_resp, history: list):
    class IsInformational(BaseModel):
        reasoning: str = Field(
            description="The reason why you assert that the given response is informational for the given medical question, or why you cannot assert it. Reasoning should be concise and contain only important points."
        )
        is_informational: bool = Field(
            description="Whether the given response is informational for the given medical question."
        )
    fmted_lines = []
    for each in history:
        fmted_lines.append(f"患者：{each['patient']}")
        fmted_lines.append(f"医生：{each['doctor']}")
    fmted_lines.append(f"患者：{q}")
    fmted_lines.append(f"System：{model_out}")
    jj = '\n'.join(fmted_lines)
    prompt = f"""{jj}

对话的前几轮真实的医患对话历史，最后一轮由system对患者进行回复。请你判断system的回复是否提供了有用信息。使用 IsUseful 的格式给出。"""
    return llm.with_structured_output(IsInformational).invoke(prompt).is_informational



import os
import sys
args = int(sys.argv[1])
tt = tt[args * 2: min((args + 1) * 2, len(tt))]
known_keywords = json.load(open("./all_keys.json"))
def extract_disease_keywords(text: str) -> list[str]:
    # Define a set of known disease keywords (expandable)
    global known_keywords
    # Find matching keywords in the text
    return [keyword for keyword in known_keywords if keyword in text]
@with_retry
def hallucinate(q, model_out, ref_resp, history: list):
    class HasHallucination(BaseModel):
        reasoning: str = Field(
            description="The reason why you assert that the given response has hallucination for the given medical question, or why you cannot assert it. Reasoning should be concise and contain only important points."
        )
        has_hallucination: bool = Field(
            description="You should judge based on the given information and your own knowledge. Whether the given response has hallucination for the given medical question."
        )

    # Format conversation history
    fmted_lines = []
    for each in history:
        fmted_lines.append(f"患者：{each['patient']}")
        fmted_lines.append(f"医生：{each['doctor']}")
    fmted_lines.append(f"患者：{q}")
    fmted_lines.append(f"System：{model_out}")

    # Extract disease keywords from the query and model output
    query_keywords = extract_disease_keywords(q)
    model_output_keywords = extract_disease_keywords(model_out)

    # Collect relevant documents
    retrieved_content = []
    for keyword in set(query_keywords + model_output_keywords):
        docs = retriver.invoke(keyword)
        retrieved_content.extend([doc.metadata['content'] for doc in docs])

    # Combine retrieved content
    combined_retrieved_content = "\n".join(retrieved_content)
    jj = '\n'.join(fmted_lines)
    # Build hallucination detection prompt
    judge_hallucination_pmpt = f"""{jj}

对话的前几轮真实的医患对话历史，最后一轮由System对患者进行回复。最后一轮医生的真实回复是

{ref_resp}

一些可能的相关信息如下：

{combined_retrieved_content}

请你基于自己的知识和提供的，可能相关的信息判断system的回复是否存在医学幻觉。回答yes 或者no"""

    # Use the model to evaluate hallucination
    chain = llm.with_structured_output(HasHallucination)
    result = chain.invoke(judge_hallucination_pmpt)
    return result.has_hallucination

from bert_score import score
from Levenshtein import ratio
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
model = AutoModel.from_pretrained(bert_model_path, device_map="auto")
model.eval()
def get_word_embedding(word_list):
    global tokenizer, model
    # 批量编码
    encoded = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
    # 获取BERT输出
    with torch.no_grad():
        outputs = model(**encoded)
    # 获取CLS向量 (取第一个位置的向量)
    # cls_embeddings = outputs.last_hidden_state[:, 0, :]    
    # return cls_embeddings

    hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
    attention_mask = encoded['attention_mask'] 
    # 去除每个序列的[CLS]和[SEP]标记
    hidden_states = hidden_states[:, 1:-1, :]  # 切片去除首尾标记
    attention_mask = attention_mask[:, 1:-1]   # 相应地调整attention mask
    
    # 计算平均值（考虑attention mask）
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    hidden_states_masked = hidden_states * mask_expanded
    sum_embeddings = torch.sum(hidden_states_masked, dim=1)  # 在序列长度维度求和
    sum_mask = torch.clamp(attention_mask.sum(dim=1).unsqueeze(-1), min=1e-9)  # 防止除零
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings

def calculate_similarity_matrix(list1, list2):
    """
    计算两个词语列表之间的相似度矩阵
        list1: 第一个词语列表
        list2: 第二个词语列表
    Returns:
        相似度矩阵，维度为 len(list1) x len(list2)
    """
    # 获取两个列表的CLS向量
    embeddings1 = get_word_embedding(list1)
    embeddings2 = get_word_embedding(list2)
    # 归一化向量
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(embeddings1, embeddings2.transpose(0, 1))
    return similarity_matrix.numpy()


def dsc_bert(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    print(similarity_matrix)
    hit_p = 0
    for i, p in enumerate(prediction):
        for j, g in enumerate(gold):
            if similarity_matrix[i, j]>=threshold:
                print(p, g)
                hit_p += 1
    return (2*hit_p)/(len(prediction)+len(gold))

def dsc_levenshtein(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >=  threshold:
                hit_p += 1
                print(p)
    return (2*hit_p)/(len(prediction)+len(gold))

def jaccard_levenshtein(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    hit_p = 0
    for p in prediction:
        for g in gold:
            if ratio(p, g) >=  threshold:
                hit_p += 1
    return hit_p/(len(prediction)-hit_p+len(gold))

def jaccard_bert(prediction, gold, threshold):
    """
    prediction: list, 模型预测的询问实体和概念
    gold: list, 医生真实的询问实体和概念
    """
    similarity_matrix = calculate_similarity_matrix(prediction, gold)
    hit_p = 0
    for i, p in enumerate(prediction):
        for j, g in enumerate(gold):
            if similarity_matrix[i, j]>=threshold:
                print(p, g)
                hit_p += 1
    return hit_p/(len(prediction)-hit_p+len(gold))


# if __name__ == '__main__':
#     prediction = ["咳嗽", "感冒", "疼痛", "伴随症状", "严重程度", "发烧"]
#     gold = ["疼", "伴随症状"]
#     print(dsc_bert(prediction, gold, 0.8))
#     print(jaccard_bert(prediction, gold, 0.8))
def flatten_med_ask_content_to_list(med_ask_content: MedicalAskContent) -> list:
    """
    Convert MedicalAskContent to a list of strings
    """
    return [
        e for e in [
            *med_ask_content.body_part,
            *med_ask_content.concept,
            *med_ask_content.disease,
            *med_ask_content.medcine,
            *med_ask_content.medical_check,
            *med_ask_content.surgery,
        ] if e is not None
    ]

for each in tt:
    if os.path.exists("./output/{}.txt".format(each)):
        continue
    if 'ask' not in each:
        continue
    print(f"processing {each}")
    data = json.load(open("./mrag/{}".format(each)))
    should_be_ask = "ask" in each
    ask = 0
    tell = 0
    lev_info = []
    auv = []
    entities = []
    from threading import Lock
    # lock = Lock()
    lock = None
    total_informational_div = 0
    informational = 0
    # list of 0, 1
    support_info = []
    br = 0
    bf = 0
    bp = 0
    b_div = 0
    dsc_berts = []
    dsc_levenshteins = []
    jaccard_levenshteins = []
    jaccard_berts = []
    def work(model_out):
        global ask, tell, lev_info, auv, lock, logger, entities
        global total_informational_div, informational, support_info
        global br, bf, bp, b_div
        global dsc_berts, dsc_levenshteins, jaccard_levenshteins, jaccard_berts
        model_is_asking = model_out['model']['is_asking'] if model_out['model'] is not None else False
        model_out['model'] = model_out['model']['response'] if model_out['model'] is not None else ""
        model_out['src']['conversation'] = (
            model_out['src']['conversation']
            if
            'conversation' in model_out['src']
            else [
                {
                    "patient": "",
                    "doctor": ""
                }
            ]
        )
        logger.info(model_out)
        if model_is_asking:
            model_entities = generate_entities(model_out["model"])
            actual_resp = model_out['src']['conversation'][-1]['doctor']
            actual_entities = generate_entities(actual_resp)
            t_auv = (1, 1)
            if (
                len(
                    flatten_med_ask_content_to_list(model_entities)
                ) == 0
                and
                len(
                    flatten_med_ask_content_to_list(actual_entities)
                ) == 0
            ):
                dsc_berts.append(1)
                dsc_levenshteins.append(1)
                jaccard_levenshteins.append(1)
                jaccard_berts.append(1)
                return
            elif bool(
                len(
                    flatten_med_ask_content_to_list(model_entities)
                ) == 0
                or
                len(
                    flatten_med_ask_content_to_list(actual_entities)
                ) == 0
            ):
                dsc_berts.append(0)
                dsc_levenshteins.append(0)
                jaccard_levenshteins.append(0)
                jaccard_berts.append(0)
                return
            lev_d = Levenshtein.ratio(
                flatten_medical_ask_content(actual_entities),
                flatten_medical_ask_content(model_entities),
            )
            dsc_berts.append(dsc_bert(
                flatten_med_ask_content_to_list(model_entities),
                flatten_med_ask_content_to_list(actual_entities),
                0.8,
            ))
            dsc_levenshteins.append(dsc_levenshtein(
                flatten_medical_ask_content(actual_entities),
                flatten_medical_ask_content(model_entities),
                0.8,
            ))
            jaccard_levenshteins.append(jaccard_levenshtein(
                flatten_medical_ask_content(actual_entities),
                flatten_medical_ask_content(model_entities),
                0.8,
            ))
            jaccard_berts.append(jaccard_bert(
                flatten_med_ask_content_to_list(model_entities),
                flatten_med_ask_content_to_list(actual_entities),
                0.8,
            ))
            entities.append(
                (actual_entities, model_entities)
            )
            ask += 1
            lev_info.append(lev_d)
            auv.append(
                t_auv
            )
        else:
            tell += 1
            actual_resp = model_out['src']['conversation'][-1]['doctor']
    futures = []
    for i in tqdm(data[:]):
        work(i)
    with open("/data/wsw/very_acc_new_new_me/output/{}.txt".format(each), "w") as f:
        f.write("\n")
        f.write(f"ask: {ask}\ntell: {tell}\n")
        import numpy as np
        f.write(f"lev distance: {np.mean(lev_info)}\n")
        f.write(f"jaccard levenshtein: {np.mean(jaccard_levenshteins)}\n")
        f.write(f"jaccard bert: {np.mean(jaccard_berts)}\n")
        f.write(f"dsc levenshtein: {np.mean(dsc_levenshteins)}\n")
        f.write(f"dsc bert: {np.mean(dsc_berts)}\n")
        # f.write(f"""informational_ratio: {informational / total_informational_div}\n""")
        # f.write(f"""informational div: {total_informational_div}\n""")
        # f.write(f"""informational_cnt: {informational}\n""")
        # f.write(f"""support_info: {np.mean(support_info)}\n""")
        # f.write(f"""bert f: {bf / b_div}\n""")
        # f.write(f"""bert r: {br / b_div}\n""")
        # f.write(f"""bert p: {bp / b_div}\n""")
        # f.write(f"""bert div: {b_div}\n""")

    def serialize(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return obj

    # with open("./mm/{}.json".format(each), "w") as f:
    #     json.dump({
    #         "support_info": support_info,
    #         "auv_info": auv,
    #         "lev_info": lev_info,
    #         "entities": entities,
    #     }, f, indent=4, default=serialize)


    # with open("./mm/{}.json".format(each), "w") as f:
    #     json.dump(data_new, f, indent=4)
