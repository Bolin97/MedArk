base_model = "Qwen/Qwen2.5-7B-Instruct"
sft_adapter = "../train-sft/out_trainer"
# data should be from the output of first_pass_gen.py
work_on_data = "./data_with_reasonings_0.pkl"
embedd_path = "moka-ai/m3e-base"
chroma_collection_name = "collection_name"
chroma_path = "../docs/chroma"

from typing import Annotated, Literal, TypedDict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import pickle
from tqdm import tqdm
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json
import uuid
from langchain_core.documents import Document
from time import time
import chromadb
from langgraph.graph.message import add_messages
from pydantic import Field, BaseModel
import loguru
import pickle
import requests as rq
from bs4 import BeautifulSoup
from urllib.parse import quote, quote_plus, urlparse, parse_qs, urlunparse, urlencode
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
import regex
import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from threading import Lock
import nest_asyncio
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
nest_asyncio.apply()
logger = loguru.logger
try:
    logger.remove(0)
except:
    pass
logger.add("a.log")
from concurrent.futures import ThreadPoolExecutor
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
class RetrievalAction(BaseModel):
    """The action of retrievaling from the knowledge base."""
    reasoning: str = Field(
        description="The reasong why you are doing the following query."
    )
    queries: list[str] = Field(description="The list of queries to execute. Each should be a question.")

class RetrievalItem(TypedDict):
    
    query: str
    refined_result: str

class AskAction(BaseModel):
    """The action of asking the patient for extra information."""
    reasoning: str = Field(
        description="The reason why you are asking the following question."
    )
    disease: list[str] = Field(
        default_factory=list,
        description="你提问的疾病对象"
    )
    symptom: list[str] = Field(
        default_factory=list,
        description="你提问的症状对象"
    )
    medcine: list[str] = Field(
        default_factory=list,
        description="你提问的药物对象"
    )
    surgery: list[str] = Field(
        default_factory=list,
        description="你提问的手术对象"
    )
    body_part: list[str] = Field(
        default_factory=list,
        description="你提问的身体部位对象"
    )
    medical_check: list[str] = Field(
        default_factory=list,
        description="你提问的检查项目对象"
    )
    concept: list[str] = Field(
        default_factory=list,
        description="你提问的问诊医学概念对象"
    )
    text: str = Field(
        description="The text sent to the patient."
    )

class TellAction(BaseModel):
    """The action of telling the patient something."""
    reasoning: str = Field(
        description="The reason why you are telling the user the following text."
    )
    text: str = Field(
        description="The text sent to the patient."
    )

class Conversation(TypedDict):
    
    patient: str
    doctor: str

class RagMeta(TypedDict):

    vector_store: list[Document]
    baidu_search: tuple[
        Annotated[str, "question"],
        Annotated[str, "answer"],
        Annotated[str, "failed"],
        Annotated[str, "src"]
    ]

class RetrievalRequest(BaseModel):

    reasoning: str = Field(
        description="The process of how you arrived on the following queries you are going to make based on the input. Reasoning should be concise and contain only important points."
    )
    queries: list[str] = Field(
        description="The queries you want to make. Each should be a question."
    )

class RefinedResult(BaseModel):

    refined_result: str = Field(
        description="The refined result of the given message. Should be in no more than 2 to 3 sentences. Should contain no more content than what's related to the query."
    )

class ReasoningCheck(BaseModel):

    reasoning: str = Field(
        description="The reason why you assert that you can deduce the answer of the professional doctor from the given information, or why you cannot deduce it. Reasoning should be concise and contain only important points."
    )
    deducible: bool = Field(description="Whether you can deduce the answer of the professional doctor from the given information.")

class Response(BaseModel):

    reasoning: str = Field(
        description="The reason why you ask the user for more information or respond to them. Reasoning should be concise and contain only important points."
    )
    is_asking: bool = Field(
        description="Wehther your sentence asks the user for more information or tell them something. If this is true, this response will be marked as ask, or else it will be marked as tell."
    )
    entities: list[str] = Field(
        description="The medical entities related to this conversation. Only the important ones should be included."
    )
    text: str = Field(
        description="The text sent to the patient."
    )

class RetrievalItem(TypedDict):
    
    query: str
    refined_result: str

class StructuredOutputWithRaw(TypedDict):
    
    raw: AIMessage
    parsed: Any
    parsing_error: Any

def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=embedd_path)
    persistent_client = chromadb.PersistentClient(path=chroma_path)
    collection = persistent_client.get_or_create_collection(chroma_collection_name)
    vector_store = Chroma(
        client=persistent_client,
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
    )
    return vector_store


def do_retrieval(request: RetrievalAction, summarizer: BaseChatModel, vector_store: Chroma) -> tuple[list[RetrievalItem], Annotated[list[Any], "rag result"]]:
    class RefinedResult(BaseModel):
        """The refined result of the given message."""
        refined_result: str = Field(
            description="The refined result of the given message. Should be in no more than 2 to 3 sentences. Should contain no more content than what's related to the query."
        )
    llm_s = summarizer.with_structured_output(RefinedResult)
    prompt = "你是一个医疗AI助手，请你从result中提炼出与患者问题最相关且贴近日常生活的信息，用于生成口语化的回复。以 RefinedResult 的形式给出。\n\nquery: {}\nresult: {}"
    document_format = """问题：{}，答案：{}"""
    res = []
    metas = []
    for q in request.queries:
        def work():
            formatted_documents = []
            TOPK = 2
            query_result = vector_store.similarity_search(
                q,
                k=TOPK
            )
            if query_result is not None:
                query_result = [
                    qr for qr in query_result
                    if any(ch in q for ch in qr.page_content)
                ]
            if query_result is not None:
                formatted_documents.extend([
                    document_format.format(d.metadata["question"], d.metadata["answer"])
                    for d in query_result
                ])
            question, answer, failed, src = [None] * 4
            # if query_result is None or len(query_result) < TOPK:
            #     question, answer, failed, src = search_baidu(q)
            # if not failed:
            #     formatted_documents.append(document_format.format(question, answer))
            p = prompt.format(q, "\n".join(formatted_documents))
            result = llm_s.invoke(p)
            return result, {
                "vector_store": query_result,
                "baidu_search": (question, answer, failed, src)
            }
        result, meta = work()
        for _ in range(3):
            try:
                result, meta = work()
                assert result is not None
                break
            except Exception as e:
                logger.error(e)
        res.append(RetrievalItem(query=q, refined_result=result.refined_result))
        metas.append(meta)
    return res, metas

def get_summarizer() -> BaseChatModel:
    return ChatOllama(
        model="qwen2.5:32b"
    )

def do_query(request: RetrievalAction, summerizer: BaseChatModel, vector_store: Chroma) -> tuple[list[RetrievalItem], Annotated[list[RagMeta], "the raw object returned by RAG"]]:
    class RefinedResult(BaseModel):
        """The refined result of the given message."""
        refined_result: str = Field(
            description="The refined result of the given message. Should be in no more than 2 to 3 sentences. Should contain no more content than what's related to the query."
        )
    llm_s = summerizer.with_structured_output(RefinedResult)
    prompt = """你是一个医疗AI助手，请你从result中提炼出与患者问题最相关且贴近日常生活的信息，用于生成口语化的回复。以 RefinedResult 的形式给出。

query: {}
result: {}
"""
    document_format = """问题：{}，答案：{}"""
    res = []
    metas = []
    for q in request.queries:
        def work():
            formatted_documents = []
            TOPK = 2
            query_result = vector_store.similarity_search(
                q,
                k=TOPK
            )
            if query_result is not None:
                query_result = [
                    qr for qr in query_result
                    if sum(1 if ch in q else 0 for ch in qr.page_content) >= 3
                ]
            if query_result is not None:
                formatted_documents.extend([
                    document_format.format(d.metadata["question"], d.metadata["answer"])
                    for d in query_result
                ])
            question, answer, failed, src = [None] * 4
            # if query_result is None or len(query_result) < TOPK:
            #     question, answer, failed, src = search_baidu(q)
            #     if not failed:
            #         formatted_documents.append(document_format.format(question, answer))
            p = prompt.format(q, "\n".join(formatted_documents))
            result = llm_s.invoke(p)
            return result, {
                "vector_store": query_result,
                "baidu_search": (question, answer, failed, src)
            }
        result, meta = work()
        for _ in range(3):
            try:
                result, meta = work()
                assert result is not None
                break
            except Exception as e:
                logger.error(e)
        res.append(RetrievalItem(query=q, refined_result=result.refined_result))
        metas.append(meta)
    return res, metas

def get_start_prompt(conv: list[Conversation]) -> str:
    formatted_conv = []
    for each in conv:
        formatted_conv.append(f"患者：{each['patient']}")
        formatted_conv.append(f"医生：{each['doctor']}")
    start_prompt = '''<|im_start|>system

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "RetrievalAction", "description": "The action of retrievaling from the knowledge base.", "parameters": {"properties": {"reasoning": {"description": "The reasong why you are doing the following query.", "type": "string"}, "queries": {"description": "The list of queries to execute. Each should be a question.", "items": {"type": "string"}, "type": "array"}}, "required": ["reasoning", "queries"], "type": "object"}}}
{"type": "function", "function": {"name": "AskAction", "description": "The action of asking the patient for extra information.", "parameters": {"properties": {"reasoning": {"description": "The reason why you are asking the following question.", "type": "string"}, "disease": {"description": "你提问的疾病对象", "items": {"type": "string"}, "type": "array"}, "symptom": {"description": "你提问的症状对象", "items": {"type": "string"}, "type": "array"}, "medcine": {"description": "你提问的药物对象", "items": {"type": "string"}, "type": "array"}, "surgery": {"description": "你提问的手术对象", "items": {"type": "string"}, "type": "array"}, "body_part": {"description": "你提问的身体部位对象", "items": {"type": "string"}, "type": "array"}, "medical_check": {"description": "你提问的检查项目对象", "items": {"type": "string"}, "type": "array"}, "concept": {"description": "你提问的问诊医学概念对象", "items": {"type": "string"}, "type": "array"}, "text": {"description": "The text sent to the patient.", "type": "string"}}, "required": ["reasoning", "text"], "type": "object"}}}
{"type": "function", "function": {"name": "TellAction", "description": "The action of telling the patient something.", "parameters": {"properties": {"reasoning": {"description": "The reason why you are telling the user the following text.", "type": "string"}, "text": {"description": "The text sent to the patient.", "type": "string"}}, "required": ["reasoning", "text"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"arguments": <args-json-object>, "name": <function-name>}
</tool_call><|im_end|>
<|im_start|>user
### 指令
是一个医学专家级的AI助手。现在你需要根据已有信息，给出下一个Action，如果信息不足，可以使用RetrievalAction查询知识库，或使用AskAction询问患者。如果信息充足，可以使用TellAction告知患者信息。注意，你一次只能执行一个Action。

### 医患历史对话
<|__CONVERSATION__|>

<|im_end|>
<|im_start|>assistant
<tool_call>
{"arguments": {"reasoning": "'''
    return start_prompt.replace("<|__CONVERSATION__|>", "\n".join(formatted_conv[:-1]))

from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast
import torch.nn.functional as F

def generate_next_action(acc: str, llm: Qwen2ForCausalLM, tokenizer: Qwen2TokenizerFast, max_length: int = 2048, force_stop_retrieval: bool = False) -> tuple[str, Literal["r"] | Literal["t"] | Literal["a"] | None, AskAction | TellAction | RetrievalAction, Annotated[str | None, "err"], list[float]]:
    inputs = tokenizer(acc, return_tensors='pt')
    inputs['input_ids'] = inputs['input_ids'].to('cuda')
    inputs['attention_mask'] = inputs['attention_mask'].to('cuda')
    from transformers import GenerationConfig
    r = llm.generate(
        **inputs,
        max_length=2048,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated_ids = r.sequences[0]  # The generated token IDs

    # Convert logits to probabilities and get scores for the generated tokens
    token_scores = []
    for i, logits in enumerate(r.scores):
        # Softmax to convert logits to probabilities
        probs = torch.nn.functional.softmax(logits[0], dim=-1)
        token_id = generated_ids[len(inputs['input_ids'][0]) + i].item()  # Skip input tokens
        token_score = probs[token_id].item()
        token_scores.append(token_score)
    a_prob = token_scores



    acc = acc + tokenizer.decode(r.sequences[0], skip_special_tokens=False).removeprefix(acc)
    if acc.endswith("<|endoftext|>"):
        acc = acc[:-len("<|endoftext|>")]
    # find the position of the last <tool_call> and </tool_call>
    start_pos = acc.rfind("<tool_call>")
    end_pos = acc.rfind("</tool_call>")
    if start_pos == -1 or end_pos == -1:
        return acc, None, None, "No tool call found"
    tool_call_str = acc[start_pos + len("<tool_call>"):end_pos]
    try:
        obj = json.loads(tool_call_str)
    except:
        return acc, None, None, "Invalid json", []
    if obj["name"] == "RetrievalAction":
        return acc, "r", RetrievalAction(
            reasoning=obj["arguments"]["reasoning"],
            queries=obj["arguments"]["queries"]
        ), None, a_prob
    elif obj["name"] == "TellAction":
        return acc, "t", TellAction(
            reasoning=obj["arguments"]["reasoning"],
            text=obj["arguments"]["text"]
        ), None, a_prob
    elif obj["name"] == "AskAction":
        return acc, "a", AskAction(
            reasoning=obj["arguments"]["reasoning"],
            disease=obj["arguments"]["disease"],
            symptom=obj["arguments"]["symptom"],
            medcine=obj["arguments"]["medcine"],
            surgery=obj["arguments"]["surgery"],
            body_part=obj["arguments"]["body_part"],
            medical_check=obj["arguments"]["medical_check"],
            concept=obj["arguments"]["concept"],
            text=obj["arguments"]["text"],
        ), None, a_prob
    return acc, None, None, "Unknown tool call", a_prob

def handle_retrieval(request: RetrievalAction, vector_store: Chroma, summerizer: BaseChatModel) -> tuple[str, list[RagMeta]]:
    res, metas = do_query(request, summerizer, vector_store)
    formatted_res = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in res
    )
    return f"""<|im_start|>user
<tool_response>
{formatted_res}
</tool_response><|im_end|>
""", metas

def generate_action_sequence(conv: list[Conversation], vector_store: Chroma, summerizer: BaseChatModel) -> tuple[list[RetrievalAction], AskAction | TellAction, Annotated[str, "acc"], list[list[RagMeta]], list[list[float]]]:
    force_tool = '''<|im_start|>assistant
<tool_call>
{"arguments": {"reasoning": "'''
    acc = get_start_prompt(conv)
    retireval_seq = []
    final = None
    cnt = 0
    rag_metas = []
    a_probs = []
    while True:

        acc, tool, action, err, a_prob = generate_next_action(acc, llm, tokenizer, cnt >= 4)
        for _ in range(5):
            if err is None:
                break
            else:
                acc, tool, action, err, a_prob = generate_next_action(acc, llm, tokenizer, cnt >= 4)
        if tool != "r":
            final = action
            break
        retireval_seq.append(action)
        tool_resp, metas = handle_retrieval(action, vector_store, summerizer)
        rag_metas.append(metas)
        acc += tool_resp
        cnt += 1
        a_probs.append(a_prob)
        if not acc.endswith('\n'):
            acc += "\n"
        acc = acc + force_tool
    return retireval_seq, final, acc, rag_metas, a_probs
    
class EvalItem(TypedDict):

    conversation: list[Conversation]
    response: Response

llm = Qwen2ForCausalLM.from_pretrained(base_model, device_map="auto")
llm.load_adapter(sft_adapter)
llm.eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(base_model, device_map="auto")
summarizer = get_summarizer()
vector_store = get_vector_store()

def evaluate(item: EvalItem) -> tuple[
    Annotated[list[RetrievalAction], "Retrievl seq"],
    Annotated[TellAction | AskAction, "Reponse action"],
    Annotated[str, "acc"],
    Annotated[int, "hit cnt"],
    Annotated[bool, "response action match"],
    list[list[float]]
]:
    retrieval_actions, repsonse_action, acc, _, a_probs = generate_action_sequence(item['conversation'], vector_store, summarizer)
    resp_action_match = (
        (
            item['response_structured_with_raw']['parsed'].is_asking 
            and 
            isinstance(repsonse_action, AskAction)
        ) or (
            not item['response_structured_with_raw']['parsed'].is_asking 
            and 
            not isinstance(repsonse_action, AskAction)
        )
    )
    hit = 0
    if isinstance(repsonse_action, AskAction):
        # hit = sum(
        #     [
        #         1 if (
        #             entity in item['conversation'][-1]['doctor']
        #         ) else 0 for entity in repsonse_action.entities
        #     ]
        # )
        hit = 0
    return retrieval_actions, repsonse_action, acc, hit, resp_action_match, a_probs

def custom_serializer(obj):
    return obj.__dict__

import torch

@torch.inference_mode()
def evaluate_all(items: EvalItem, raw: list[any]):
    total_hit = 0
    hit_divider = 0
    src_data_ask_times = 0
    total_resp_action_match = 0
    total_resp_action_divider = 0
    result_folder = "./result"
    this_model_asked = 0
    this_model_told = 0
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(result_folder + "/acc", exist_ok=True)
    os.makedirs(result_folder + "/retri", exist_ok=True)

    records = []
    valid_records = []

    def work(idx, item):

        nonlocal src_data_ask_times, total_hit, hit_divider, total_resp_action_match, total_resp_action_divider
        nonlocal records
        retrieval_actions, repsonse_action, acc, hit, resp_action_match, a_probs = evaluate(item)
        records.append(
            [item, retrieval_actions, repsonse_action, a_probs]
        )
        nonlocal this_model_asked, this_model_told
        if isinstance(repsonse_action, AskAction):
            this_model_asked += 1
        else:
            this_model_told += 1
        with open(f"{result_folder}/acc/{idx}.txt", "w", encoding='utf-8') as f:
            f.write(acc)
            f.write("\n\n" + "=" * 10 + "\n\n")
            f.write(
                json.dumps(
                    raw[idx], default=custom_serializer, indent=2, ensure_ascii=False
                )
            )
        with open(f"{result_folder}/retri/{idx}.json", "w", encoding='utf-8') as f:
            json.dump([e.__dict__ for e in retrieval_actions], f, ensure_ascii=False)
        total_resp_action_match += 1 if resp_action_match else 0
        total_resp_action_divider += 1
        if isinstance(repsonse_action, AskAction):
            if not os.path.exists("./aaa.txt"):
                open("./aaa.txt", "w").close()
            with open("./aaa.txt", "a") as f:
                f.write(f"\n{idx}\n")
        if item['response_structured_with_raw']['parsed'].is_asking:
            src_data_ask_times += 1
            if resp_action_match and isinstance(repsonse_action, AskAction):
                total_hit += hit
                hit_divider += 1
                if not os.path.exists("./fff.txt"):
                    open("./fff.txt", "w").close()
                with open("./fff.txt", "a") as f:
                    f.write(f"\n{idx}\n")

    # with ThreadPoolExecutor(max_workers=16) as executor:
    for idx, item in tqdm(enumerate(items), position=1, total=len(items)):
        
        while True:
            try:
                work(idx, item)
            except:
                print("ERR")
                continue
            break
    print(records[0])
    pickle.dump(records, open("./records.pkl", "wb"))
    with open(f"{result_folder}/r.txt", "w", encoding='utf-8') as f:
        if hit_divider != 0:
            f.write(f"acc: {total_hit / hit_divider}\n")
    with open(f"{result_folder}/r.txt", "a", encoding='utf-8') as f:
        if total_resp_action_divider != 0:
            f.write(f"resp_action_match: {total_resp_action_match / total_resp_action_divider}\n")
    with open(f"{result_folder}/r.txt", "a", encoding='utf-8') as f:
        f.write(f"hit number: {total_hit}\n")
    with open(f"{result_folder}/r.txt", "a", encoding='utf-8') as f:
        f.write(f"ask match: {hit_divider}\n")
    with open(f"{result_folder}/r.txt", "a", encoding='utf-8') as f:
        f.write(f"ask times: {src_data_ask_times}\n")
    with open(f"{result_folder}/r.txt", "a", encoding='utf-8') as f:
        if src_data_ask_times != 0:
            f.write(f"ask match rate: {total_hit / src_data_ask_times}\n")
    with open(f"{result_folder}/r.txt", "w", encoding='utf-8') as f:
        f.write(f"\n")
    with open(f"{result_folder}/tb.txt", "w", encoding='utf-8') as f:
        f.write(f"this model asked: {this_model_asked}\n")
    with open(f"{result_folder}/tb.txt", "a", encoding='utf-8') as f:
        f.write(f"this model told: {this_model_told}\n")

ld = pickle.load(open("./bal.pkl", "rb"))
items = [
    {
        "conversation": l['conversation'],
        "response_structured_with_raw": l['response_structured_with_raw'],
        "all_meta": l
    } for l in ld[:]
]
evaluate_all(items, ld)
# generate_action_sequence([{"patient":"男放有尿道炎对宝宝有没有影响", "doctor": ""}], vector_store, summarizer)
# driver.quit()