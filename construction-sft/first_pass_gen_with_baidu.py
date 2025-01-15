openai_api_key = ''
baidu_cookie_1 = ''
baidu_cookie_2 = ''
embedd_path = "moka-ai/m3e-base"
chroma_collection_name = "collection_name"
chroma_path = "../docs/chroma"
src_data = [
    [
        {
            "doctor": "example",
            "patient": "example"
        },
        {
            "doctor": "example",
            "patient": "example"
        }
    ],
    [
        {
            "doctor": "example",
            "patient": "example"
        },
        {
            "doctor": "example",
            "patient": "example"
        }
    ]
]

from typing import Annotated, Literal, TypedDict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
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
from langchain_community.llms.vllm import VLLM
nest_asyncio.apply()
import os
import time
import requests
import json

# In[2]:


logger = loguru.logger
try:
    logger.remove(0)
except:
    pass
logger.add("a.log")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.openai-proxy.org/v1",
    api_key=openai_api_key
)
llm_local = ChatOllama(model="qwen2.5:32b")

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

def get_page_by_broswer(link: str) -> str:
    if not browser_lock.acquire(timeout=120):
        return ""
    try:
        driver.get(link)
        r = driver.page_source
        return r
    finally:
        browser_lock.release()

headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'User-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.62",
    'Cookie': baidu_cookie_1,
    'Host': 'www.baidu.com',
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Referer': 'https://www.baidu.com/',
}
another_heder = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'User-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0",
    'Cookie': baidu_cookie_2,
    'Host': 'www.baidu.com',
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Referer': 'https://www.baidu.com/',
}

baidu_lock = Lock()
def search_baidu(question: str) -> tuple[
    Annotated[str, "question"], 
    Annotated[str, "answer"], 
    Annotated[str | None, "error"],
    Annotated[str, "source"]
]:
    search_result = None
    global baidu_lock
    if not baidu_lock.acquire(timeout=120):
        return "", "", "LOCKED", ""
    try:
        if False:
            search_result = get_page_by_broswer(f"http://www.baidu.com/s?wd={question}")
        else:
            c = random.randint(0, 3)
            search_result_pg = None
            if c == 0:
                search_result_pg = rq.get(
                    f"https://www.baidu.com/s?wd="+question,
                    headers=headers,
                )
            elif c == 1:
                search_result_pg = rq.get(
                    f"https://www.baidu.com/s?wd="+question,
                    headers=another_heder,
                )
            elif c == 2:
                search_result_pg = rq.get(
                    f"https://www.baidu.com/s?wd="+question,
                    headers={
                        **headers,
                        'Cookie': ''
                    },
                )
            elif c == 3:
                search_result_pg = rq.get(
                    f"https://www.baidu.com/s?wd="+question,
                    headers={
                        **another_heder,
                        'Cookie': ''
                    },
                )
            search_result = search_result_pg.text
    except:
        pass
    finally:
        baidu_lock.release()
    return_link = ""
    def try_qa() -> tuple[
        Annotated[str, "question"], 
        Annotated[str, "answer"], 
        Annotated[str | None, "error"]
    ]:
        search_soup = BeautifulSoup(search_result, 'html.parser')
        if "网络不给力，请稍后重试" in search_result:
            logger.info("ANTI CRAWLING TRIGGERED")
            # time.sleep((float(random.randint(0, 1000)) / 1000) * 15)
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

embeddings = HuggingFaceEmbeddings(model_name=embedd_path)
persistent_client = chromadb.PersistentClient(path=chroma_path)
collection = persistent_client.get_or_create_collection(chroma_collection_name)
vector_store = Chroma(
    client=persistent_client,
    collection_name=chroma_collection_name,
    embedding_function=embeddings,
)

class RetrievalRequest(BaseModel):
    """Make a request to query the knowledge base."""
    reasoning: str = Field(
        description="The process of how you arrived on the following queries you are going to make based on the input. Reasoning should be concise and contain only important points."
    )
    queries: list[str] = Field(
        description="This should be the queries you want to make. Each should be a question."
    )

class RefinedResult(BaseModel):
    """The refined result of the given message."""
    refined_result: str = Field(
        description="The refined result of the given message. Should be in no more than 2 to 3 sentences. Should contain no more content than what's related to the query."
    )

class ReasoningCheck(BaseModel):
    """Decide wether you can deduce the answer of the professional doctor from the given information."""
    reasoning: str = Field(
        description="The reason why you assert that you can deduce the answer of the professional doctor from the given information, or why you cannot deduce it. Reasoning should be concise and contain only important points."
    )
    deducible: bool = Field(description="Whether you can deduce the answer of the professional doctor from the given information.")

class Response(BaseModel):
    """A response to the user."""
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

class Conversation(TypedDict):
    
    patient: str
    doctor: str

class RetrievalItem(TypedDict):
    
    query: str
    refined_result: str

class StructuredOutputWithRaw(TypedDict):
    
    raw: AIMessage
    parsed: Any
    parsing_error: Any

def get_query_request(conv: list[Conversation], query_requests: list[RetrievalRequest], query_history_by_turn: list[list[RetrievalItem]] = None) -> StructuredOutputWithRaw:
    conversation_history_formatted = []
    for c in conv:
        conversation_history_formatted.append(
            f"患者：{c['patient']}"
        )
        conversation_history_formatted.append(
            f"医生：{c['doctor']}"
        )
    query_history_formatted = [
        "\n".join([
            f"- query: {q['query']}\n- result: {q['refined_result']}" for q in turn
        ]) for turn in query_history_by_turn
    ]
    query_history_turns = "\n".join(
        f"第 {i} 轮：\n{content}" for i, content in enumerate(query_history_formatted)
    )
    
    llm_s = llm.with_structured_output(RetrievalRequest, include_raw=True)
    conv_hist = '\n'.join(conversation_history_formatted[:-1])
    p = f"""你是一个基于外部知识库的医学专家级的AI助手。现在给你一段医患历史对话和历史检索结果，你的任务是拆解患者问题并生成一组中文查询子句。如果患者表述中有不专业医学术语或者错别字，请在查询子句中更正过来。

### 医患历史对话
{conv_hist}

### 历史检索结果
当检索还未开始时，该历史信息为空。检索开始后，每次的返回结果会追加到历史检索结果中。

{query_history_turns}

给出你新的检索请求，你的请求应该少而精准，每次不能超过两个query。注意，你不能重复之前的查询结果。
"""

    res = llm_s.invoke(
        p
    )
    return res

def do_query(request: RetrievalRequest) -> tuple[list[RetrievalItem], Annotated[list[Any], "the raw object returned by RAG"]]:
    llm_s = llm_local.with_structured_output(RefinedResult)
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
            query_result = vector_store.similarity_search(
                q,
                k=2
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

            if query_result is None or len(query_result) < 2:
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


# In[12]:


def check_duduction(conv: list[Conversation], query_request_history: list[RetrievalRequest], query_history_by_turn: list[list[RetrievalItem]] = None) -> bool:
    conversation_history_formatted = []
    for c in conv:
        conversation_history_formatted.append(
            f"患者：{c['patient']}"
        )
        conversation_history_formatted.append(
            f"医生：{c['doctor']}"
        )
    qrh_fmt = []
    for qr in query_request_history:
        queries = ',\n'.join(qr.queries)
        qrh_fmt.append(f"""{{
  "reasoning": "{qr.reasoning}",
  "action": "retrieval",
  "queries": [
    {queries}
  ]
}}""")
    qh = []
    for i in range(len(query_request_history)):
        qh.append(qrh_fmt[i])
        query_results = [
            f"""{{\n  "query": "{q['query']}",\n  "refined_result": "{q['refined_result']}"\n}}""" for q in query_history_by_turn[i]
        ]
        qh.append("<tool_response>\n")
        qh.append("\n".join(query_results))
        qh.append("\n</tool_response>")
    llm_s = llm.with_structured_output(ReasoningCheck)
    p = """你是一名专业医生。现在给你一段医患历史对话和专业医生回复，请你判断是否可以根据历史检索结果得到医生回复。以 ReasoningCheck 的形式给出。

### 医患历史对话
{}

### 专业医生回复
{}

### 历史推理路径
{}

""".format(
    "\n".join(conversation_history_formatted[:-1]),
    conversation_history_formatted[-1].replace("医生：", ""),
    "\n".join(qh)
)
    res = None
    for _ in range(3):
        if res is None:
            res = llm_s.invoke(p)
        else:
            break
    return res.deducible


# In[13]:


def work_on_one_conversation(conv: list[Conversation]) -> tuple[
    Annotated[list[StructuredOutputWithRaw], "parsed is of type QueryRequest"],
    list[RetrievalItem],
    list[list[AIMessage]],
    list[AIMessage],
    str,
    Annotated[list[list[Any]], "metadata of rag"]
]:
    query_requests = []
    query_history = []
    query_history_by_turn = []
    query_messages = []
    query_metas_by_turn = []
    cnt = 0
    break_because = ""
    while True:
        rq = get_query_request(conv, query_requests, query_history_by_turn)
        for _ in range(3):
            if rq is None or rq["parsed"] is None:
                break
            rq = get_query_request(conv, query_requests, query_history_by_turn)
        if rq is None or rq["parsed"] is None:
            break_because = "no new query_request"
            break
        query_messages.append(rq["raw"])
        query_requests.append(rq["parsed"])
        # logger.info(rq)
        query_result, metas = do_query(rq["parsed"])
        query_history.extend(query_result)
        query_metas_by_turn.append(metas)
        query_history_by_turn.append(query_result)
        done = check_duduction(conv, query_requests, query_history_by_turn)
        if done:
            break_because = "deducible"
            break
        cnt += 1
        if cnt >= 2:
            break_because = "too many retrivals"
            break
    return query_messages, query_requests, query_history_by_turn, query_history, break_because, query_metas_by_turn


# In[14]:


def get_reponse(conv: list[Conversation], query_request_history: list[RetrievalRequest], query_history_by_turn: list[list[RetrievalItem]] = None) -> StructuredOutputWithRaw:
    conversation_history_formatted = []
    for c in conv:
        conversation_history_formatted.append(
            f"患者：{c['patient']}"
        )
        conversation_history_formatted.append(
            f"医生：{c['doctor']}"
        )
    qrh_fmt = []
    for qr in query_request_history:
        queries = ',\n'.join(qr.queries)
        qrh_fmt.append(f"""{{
  "reasoning": "{qr.reasoning}",
  "action": "retrieval",
  "queries": [
    {queries}
  ]
}}""")
    qh = []
    for i in range(len(query_request_history)):
        qh.append(qrh_fmt[i])
        query_results = [
            f"""{{\n  "query": "{q['query']}",\n  "refined_result": "{q['refined_result']}"\n}}""" for q in query_history_by_turn[i]
        ]
        qh.append("<tool_response>\n")
        qh.append("\n".join(query_results))
        qh.append("\n</tool_response>")
    class IsAsking(BaseModel):
        """Wether the doctor is asking something"""
        is_asking: bool = Field(
            description="Wether the doctor is asking something",
        )
    llm_local_is_asking = llm_local.with_structured_output(IsAsking)
    def check_is_asking():
        r = llm_local_is_asking.invoke("""### 指令
你是一个智能AI助手，现在你需要判断以下给出的句子是否是问句，并以IsAsking的格式给出。

### 待判断句子
{}
""".format(conversation_history_formatted[-1].replace("医生：", "")))
        return r
    ia = check_is_asking()
    for _ in range(3):
        if ia is None:
            ia = check_is_asking()
        else:
            break
    notice = "注意，尽管专业医生的回复未知，但是根据我们收集的信息，它{}应当是一个问句。因此is_asking为{}，而且{}。".format(
        "" if not ia.is_asking else "不",
        "True" if ia.is_asking else "False",
        "你应该询问患者额外的信息" if ia.is_asking else "你应该回答患者的问题"
    )
    llm_s = llm.with_structured_output(Response, include_raw=True)
    p = """你是一个医学专家级的AI助手，能够逐步解释专业医生的推理过程。

现在，你应该解释医生如何回复，并给出推理过程以及其它相关信息。以 Response 结构的方式给出。

### 医患历史对话
{}

### 专业医生的回复
未知

### 历史推理路径
{}

### 提示
{}

""".format(
    "\n".join(conversation_history_formatted[:-1]),
    "\n".join(qh),
    notice,
)
    resp = llm_s.invoke(p)
    return resp


# In[15]:


def things_to_save(conv: list[Conversation]) -> dict:
    query_messages, query_requests, query_history_by_turn, query_history, break_because, metas = work_on_one_conversation(
        conv
    )
    resp = get_reponse(conv, query_requests, query_history_by_turn)
    return {
        "conversation": conv,
        "query_messages": query_messages,
        "query_requests": query_requests,
        "query_history": query_history,
        "query_history_by_turn": query_history_by_turn,
        "break_because": break_because,
        "response": resp["raw"],
        "response_structured_with_raw": resp,
        "rag_metas": metas
    }

# def read_into_conversations(d_path) -> list[Conversation]:
#     from datasets import load_from_disk
#     import copy
#     from tqdm import tqdm
#     d = load_from_disk(d_path)
#     conversations = []
#     for i in tqdm(range(len(d))):
#         current_conversation = []
#         for j in range(30):
#             p_idx = j * 2
#             d_idx = j * 2 + 1
#             if str(p_idx) not in d[i] or str(d_idx) not in d[i]:
#                 continue
#             if d[i][str(p_idx)] is None or d[i][str(d_idx)] is None:
#                 break
#             if str(d[i][str(p_idx)]['role']).lower() not in ["patient", "patients"]:
#                 continue
#             current_conversation.append(
#                 {
#                     "patient": d[i][str(p_idx)]['sentence'],
#                     "doctor": d[i][str(d_idx)]['sentence']
#                 }
#             )
#             conversations.append(
#                 copy.deepcopy(current_conversation)
#             )
#     return conversations

def ai_message_serializer(obj):
    if isinstance(obj, AIMessage):
        return {
            "type": "ai_message",
            "content": obj.content,
            "metadata": obj.additional_kwargs
        }
    raise TypeError(f"Type {type(obj)} not serializable")


class AIMessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AIMessage):
            return {
                "type": "ai_message",
                "content": obj.content,
                "metadata": obj.additional_kwargs,
            }
        return super().default(obj)


def default_serializer(o):
    if hasattr(o, '__dict__'):
        return o.__dict__
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

things = []
failed_conv = []
def work(c) -> tuple[dict, bool]:
    r = None
    logger.info("work_cnt")
    global failed_conv, things
    try:
        r = things_to_save(c)
    except:
        return c, True
    finally:
        logger.info("work_cnt_done")
    if r is None:
        return c, True
    return r, False

import time

max_workers = 8

pieces = 1
piece_len = int(len(src_data) / pieces)
this_piece = 0
this_piece_data = src_data[this_piece * piece_len: min((this_piece + 1) * piece_len, len(src_data))]

ckpt_times = 1000
ckpt_interval = int(piece_len / ckpt_times)

if not os.path.exists("./ckpt"):
    os.mkdir("./ckpt", 0o777)

print("total length: {}".format(len(src_data)))
print("this piece length: {}".format(len(this_piece_data)))
print("ckpt every {}".format(ckpt_interval))

resume_ckpt = 65000
things = pickle.load(open(f"./ckpt/things_{resume_ckpt}.pkl", "rb"))
failed_conv = pickle.load(open(f"./ckpt/failed_conv_{resume_ckpt}.pkl", "rb"))

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    print("submitting")
    for i, c in enumerate(this_piece_data):
        if i < resume_ckpt:
            continue
        futures.append(executor.submit(work, c))
    print("submitted")
    for idx, future in enumerate(futures):
        idx = idx + resume_ckpt
        logger.info(idx)
        print(idx)
        if idx == len(futures):
            pickle.dump(things, open("./ckpt/things_{}.pkl".format(idx), "wb"))
            pickle.dump(failed_conv, open("./ckpt/failed_conv_{}.pkl".format(idx), "wb"))
        if idx % ckpt_interval == 0:
            pickle.dump(things, open("./ckpt/things_{}.pkl".format(idx), "wb"))
            pickle.dump(failed_conv, open("./ckpt/failed_conv_{}.pkl".format(idx), "wb"))
        r, failed = future.result()
        if failed:
            r, retry_failed = work(c)
            if retry_failed:
                failed_conv.append(c)
                continue
            else:
                things.append(r)
                continue
        things.append(r)

pickle.dump(things, open(f"./data_with_reasonings_{this_piece}.pkl", "wb"))
pickle.dump(failed_conv, open(f"./failed_conv_{this_piece}.pkl", "wb"))
driver.quit()
