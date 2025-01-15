baidu_cookie = ''
base_llm = 'Qwen/Qwen2.5-7B-Instruct'
adapter = '../train-sft/out_trainer'
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
import random
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
    embeddings = HuggingFaceEmbeddings(model_name=embedd_path, model_kwargs = {'device': 'cuda:0'})
    persistent_client = chromadb.PersistentClient(path=chroma_path)
    collection = persistent_client.get_or_create_collection(chroma_collection_name)
    vector_store = Chroma(
        client=persistent_client,
        collection_name=chroma_collection_name,
        embedding_function=embeddings,
    )
    return vector_store

def setup_web_broswer() -> tuple[Lock, webdriver.Chrome]:
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式
    chrome_options.add_argument("--disable-gpu")  # 禁用 GPU，加速无头模式（可选）
    chrome_options.add_argument("--window-size=1920,1080")  # 设置窗口大小
    service = Service("/usr/local/bin/chromedriver")
    browser_lock = Lock()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get("https://www.baidu.com")
    driver.add_cookie(
        {
            "name": "BDUSS",
            "value": "FAtMUlmc01GYlZndmdLUkZOOTloSzJicGQ4emVzSzV5flFSSmxDekU0MzIzR2RuRVFBQUFBJCQAAAAAAQAAAAEAAAAhBOloWmVuZF9OaWhpbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPZPQGf2T0BnU"
        }
    )
    driver.add_cookie(
        {
            "name": "BIDUPSID",
            "value": "12BAC23F30F0C449C5EC484A7DFFDF2D"
        }
    )
    driver.add_cookie(
        {
            "name": "H_PS_PSSID",
            "value": "61027_61055_61097_60851_61129_61128_61114_61141_61218_61224_61207_61210_61208_61215_61240"
        }
    )
    return browser_lock, driver
print("SETTING UP BROWER")
browser_lock, driver = setup_web_broswer()
print("DONE")
def get_page_by_broswer(link: str) -> str:
    browser_lock.acquire()
    try:
        driver.get(link)
        r = driver.page_source
        return r
    finally:
        browser_lock.release()

def search_baidu(question: str) -> tuple[
    Annotated[str, "question"], 
    Annotated[str, "answer"], 
    Annotated[str | None, "error"],
    Annotated[str, "source"]
]:
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'User-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.62",
        'Cookie': baidu_cookie,
        'Host': 'www.baidu.com',
        'Accept': '*/*',
        'Accept-Language': 'en-GB,en;q=0.9',
        'Referer': 'https://www.baidu.com/',
    }
    rd = random.randint(0, 9)
    search_result = None
    if rd >= 6:
        search_result = get_page_by_broswer(f"http://www.baidu.com/s?wd={question}")
    else:
        search_result_pg = rq.get(
            f"https://www.baidu.com/s?wd="+question,
            headers=headers,
        )
        search_result = search_result_pg.text
    return_link = ""
    def try_qa() -> tuple[
        Annotated[str, "question"], 
        Annotated[str, "answer"], 
        Annotated[str | None, "error"]
    ]:
        search_soup = BeautifulSoup(search_result, 'html.parser')
        if "网络不给力，请稍后重试" in search_result:
            logger.info("ANTI CRAWLING TRIGGERED")
            time.sleep((float(random.randint(0, 1000)) / 1000) * 15)
            return "", "", "BLOCKED"
        result_op_divs = search_soup.find_all('div', class_="result-op")
        if result_op_divs is None:
            return "", "", "No search result div found"
        link = None
        for div in result_op_divs:
            if link is not None:
                break
            box_wrapper_div = div.find('div', id="boxWrapper")
            if box_wrapper_div is None:
                continue
            a_tags = box_wrapper_div.find_all('a')
            for a_tag in a_tags:
                if "更多" in str(a_tag) and "href" in a_tag.attrs:
                    link = a_tag["href"]
                    break
        if link is None:
            return "", "", "No search result found"
        nonlocal return_link
        return_link = link
        try:
            qa_page_text = get_page_by_broswer(link)
            qa_soup = BeautifulSoup(qa_page_text, 'html.parser')
            question = qa_soup.find('title').text
            spread_fold_div = qa_soup.find('div', id='spread-fold')
            if spread_fold_div is None:
                spread_fold_div = qa_soup
            answers = []
            ps = spread_fold_div.find_all('p')
            for p in ps:
                if p.text.strip() != "":
                    answers.append(p.text.strip())
            if len(ps) == 0:
                divs_with_content = spread_fold_div.find_all('div', "index_richTextPopup___6yTD")
                for div in divs_with_content:
                    answers.append(div.text.strip())
                if len(divs_with_content) == 0:
                    answers.append(spread_fold_div.text)
            return question, "\n".join(answers), None
        except:
            return "", "", "Error happend when parsing"
    def try_dict() -> tuple[
        Annotated[str, "question"], 
        Annotated[str, "answer"], 
        Annotated[str | None, "error"]
    ]:
        search_soup = BeautifulSoup(search_result, 'html.parser')
        if "网络不给力，请稍后重试" in search_result:
            logger.info("ANTI CRAWLING TRIGGERED")
            return "", "", "BLOCKED"
        result_divs = search_soup.find_all('div', class_="result-op")
        if len(result_divs) == 0:
            return "", "", "No search result div found"
        link = None
        for div in result_divs:
            if link is not None:
                break
            a_tags = div.find_all('a')
            for a_tag in a_tags:
                if "查看详情" in str(a_tag) and "href" in a_tag.attrs:
                    link = a_tag["href"]
                    break
        if not link:
            return "", "", "No search result found"
        nonlocal return_link
        return_link = link
        try:
            dict_redirect_page = rq.get(link, headers=headers)
            dict_page = get_page_by_broswer(dict_redirect_page.url)
            dict_soup = BeautifulSoup(dict_page, 'html.parser')
            container = dict_soup.find('div', id='richTextContainer')
            if container is None:
                container = dict_soup
            return question, container.text, None
        except Exception as e:
            return "", "", "Error happend when parsing"
    qa_res = try_qa()
    if qa_res[2] is None:
        return *qa_res, f"<|src|>baidu_qa<|search|>{question}<|link|>{return_link}"
    keep_return_link = return_link
    dict_res = try_dict()
    if dict_res[2] is None:
        return *dict_res, f"<|src|>baidu_med_dict<|search|>{question}<|link|>{return_link}"
    if len(keep_return_link) != 0:
        return *qa_res, f"<|src|>baidu_qa<|search|>{question}<|link|>{keep_return_link}"
    return *dict_res, f"<|src|>baidu_med_dict<|search|>{question}<|link|>{return_link}" 


def do_retrieval(request: RetrievalAction, summarizer: BaseChatModel, vector_store: Chroma, always_search = True) -> tuple[list[RetrievalItem], Annotated[list[Any], "rag result"]]:
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
            if always_search or query_result is None or len(query_result) < TOPK:
                question, answer, failed, src = search_baidu(q)
            if not failed:
                formatted_documents.append(document_format.format(question, answer))
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
            if query_result is None or len(query_result) < TOPK:
                question, answer, failed, src = search_baidu(q)
                if not failed:
                    formatted_documents.append(document_format.format(question, answer))
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
    # generate the only next token
    # if force_stop_retrieval:
    #     next_token = llm.generate(acc, lora_request=lora, sampling_params=SamplingParams(
    #         max_tokens=1,
    #         logit_bias={49: -1e9}
    #     ))
    #     acc = acc + next_token[0].outputs[0].text
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
    logits = r.scores
    first_token_logits = logits[0].squeeze()  # logits for the first generated token

    # Convert logits to probabilities (optional)
    # first_token_probs = F.softmax(first_token_logits, dim=-1)  # Optional: convert logits to probs
    first_token_logprobs = F.log_softmax(first_token_logits, dim=-1)  # Log probabilities
    a_prob = first_token_logprobs.detach().cpu().numpy().tolist()


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
        return acc, "r", RetrievalAction(**obj["arguments"]), None, a_prob
    elif obj["name"] == "TellAction":
        return acc, "t", TellAction(**obj["arguments"]), None, a_prob
    elif obj["name"] == "AskAction":
        return acc, "a", AskAction(**obj["arguments"]), None, a_prob
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

llm = Qwen2ForCausalLM.from_pretrained(base_llm, device_map="auto")
llm.load_adapter(adapter)

llm.eval()
tokenizer = Qwen2TokenizerFast.from_pretrained(base_llm, device_map="auto")
summarizer = get_summarizer()
vector_store = get_vector_store()

import gradio as gr
import torch

@torch.inference_mode()
def process_conversation(message, history):
    conversation = [
       {
           "patient": each_history[0],
           "doctor": each_history[1],
       } for each_history in history
    ]
    conversation.append({
        "patient": message,
        "doctor": "",
    })
    final = None
    while final is None:
        retireval_seq, final, acc, rag_metas, a_probs = generate_action_sequence(conversation, vector_store, summarizer)
    # # 模拟生成的结果
    # action_sequence = f"Action Sequence:\nStep 1: Action for {message}\nStep 2: Another action."
    # retrieval_meta = f"Retrieval Meta:\nMeta information for {message}."
    # refined_retrieval = f"Refined Retrieval:\nRefined data for {message}."
    
    # # 更新聊天记录
    # response = f"Processed: {message}"
    # history = history + [(message, response)]
    response_rr = final.text
    history = history + [(message, response_rr)]
    divier = "\n" + "=" * 5 + "\n"
    action_sequence = (divier).join(
        str(r) for r in retireval_seq
    )
    retrieval_meta = (divier).join(
        str(r) for r in rag_metas
    )
    refined_retrieval = acc
    
    return history, action_sequence, retrieval_meta, refined_retrieval

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## MedArk-3B Online Demo")
    
    # 聊天界面
    with gr.Row():
        with gr.Column(scale=2):
            chat_history = gr.Chatbot(label="Chat Interface")
            chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here.", value="老妈突然间肚子疼，疼的眼前一片黑，不敢睁眼，也不敢喘气，出冷汗，以前也发生过上次伴有呕吐，这是怎么回事？")
            send_button = gr.Button("Send")
        
        with gr.Column(scale=1):
            action_sequence_output = gr.Textbox(
                label="Query Action Sequence",
                placeholder="Query action sequence details will appear here.",
                lines=10,
                interactive=False
            )
            retrieval_meta_output = gr.Textbox(
                label="Retrieved data",
                placeholder="Retrieved data will appear here.",
                lines=10,
                interactive=False
            )
            refined_retrieval_output = gr.Textbox(
                label="Raw",
                placeholder="Raw generated text will appear here.",
                lines=10,
                interactive=False
            )
    
    # 回调逻辑
    def on_send(message, history):
        
        history, action_sequence, retrieval_meta, refined_retrieval = process_conversation(message, history)
        return history, "", action_sequence, retrieval_meta, refined_retrieval

    # 绑定回调
    send_button.click(
        fn=on_send,
        inputs=[chat_input, chat_history],
        outputs=[
            chat_history,
            chat_input,
            action_sequence_output,
            retrieval_meta_output,
            refined_retrieval_output,
        ]
    )

try:
    demo.launch(share=True, server_port=5555)
finally:
    driver.quit()