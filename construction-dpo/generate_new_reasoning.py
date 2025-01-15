# Should be the output of first_pass_gen.py after running it on the dpo data
work_on_data = "./dpo_data.pkl"
openai_api_key = ""

from typing import Annotated, Literal, TypedDict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import pandas as pd
import pickle
from tqdm import tqdm
import os
import json
from langchain_core.documents import Document
from pydantic import Field, BaseModel
import pickle
from langchain_openai import ChatOpenAI
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


# In[2]:

dpo_ld = pickle.load(open(work_on_data, "rb"))


# In[3]:


len(dpo_ld)


# In[4]:


dpo_ld[0]


# In[5]:


pmp = """<|im_start|>system

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "RetrievalAction", "description": "The action of retrievaling from the knowledge base.", "parameters": {"properties": {"reasoning": {"description": "The reasong why you are doing the following query.", "type": "string"}, "queries": {"description": "The list of queries to execute. Each should be a question.", "items": {"type": "string"}, "type": "array"}}, "required": ["reasoning", "queries"], "type": "object"}}}
{"type": "function", "function": {"name": "AskAction", "description": "The action of asking the patient for extra information.", "parameters": {"properties": {"text": {"description": "The text sent to the patient.", "type": "string"}, "reasoning": {"description": "The reason why you are asking the following question.", "type": "string"}}, "required": ["reasoning", "text"], "type": "object"}}}
{"type": "function", "function": {"name": "TellAction", "description": "The action of telling the patient something.", "parameters": {"properties": {"text": {"description": "The text sent to the patient.", "type": "string"}, "reasoning": {"description": "The reason why you are telling the user the following text.", "type": "string"}}, "required": ["reasoning", "text"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
### 指令
是一个医学专家级的AI助手。现在你需要根据已有信息，给出下一个Action的reasoning。注意，你必须完成给出的json对象，并以</tool_call>结束。

注意，你给出的 JSON 对象字符串中，禁止出现任何形式的引号。

现在，我将想你提供所有的信息。在回复中，你必须直接继续我给出的，未完成的 JSON 对象。

### 医患历史对话
[|__CONVERSATION__|]

<|im_end|>"""


# In[6]:


a_query_and_resp = """<|im_start|>assistant
<tool_call>
{"name": "RetrievalAction", "arguments": {"reasoning": "[|__REQUEST_REASONING__|]", "queries": [|__QUERIES__|]}}
</tool_call><|im_end|><|im_start|>user
<tool_response>
[|__TOOL_RESP__|]
</tool_response><|im_end|>"""


# In[7]:


each_resp = """{"query": "[|__QRY__|]", "refined_result": "[|__R_RES__|]"}"""


# In[8]:


inducer_ask = """<|im_start|>assistant
<tool_call>
{"name": "AskAction", "arguments": {"text": "[|__text__|]", "reasoning": """ + "\""
inducer_tell = """<|im_start|>assistant
<tool_call>
{"name": "AskAction", "arguments": {"text": "[|__text__|]", "reasoning": """ + "\""


# In[9]:


llm = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
    temperature=0.1
)


# In[10]:


import copy


# In[11]:


doc_reasonings = []


# In[ ]:

import loguru
logger = loguru.logger
logger.remove(0)
logger.add("logs.log", level="INFO")
for idx, each in tqdm(enumerate(dpo_ld), total=len(dpo_ld)):
    logger.info(idx)
    convs = []
    last_doc = each['conversation'][-1]['doctor'].replace('\"', '')
    for cc in each['conversation']:
        p_r = cc['patient'].replace('"', '')
        d_r = cc['doctor'].replace('"', '')
        convs.append(f"患者：{p_r}")
        convs.append(f"医生：{d_r}")
    this_pmp = pmp.replace("[|__CONVERSATION__|]", "\n".join(convs[:-1]))
    acc_retrivals = []
    for i in range(len(each['query_requests'])):
        this_retri = a_query_and_resp\
        .replace("[|__REQUEST_REASONING__|]", each['query_requests'][i].reasoning)\
        .replace("[|__QUERIES__|]", json.dumps(each['query_requests'][i].queries, ensure_ascii=False))
        this_retri_res = []
        for each_retri_res in each['query_history_by_turn'][i]:
            this_retri_res.append(
                each_resp.replace("[|__QRY__|]", each_retri_res['query']).replace("[|__R_RES__|]", each_retri_res['refined_result'])
            )
        this_retri = this_retri.replace("[|__TOOL_RESP__|]", "\n".join(this_retri_res))
        acc_retrivals.append(this_retri)
    this_pmp += "\n" + "\n".join(acc_retrivals)
    no_inducer = copy.deepcopy(this_pmp)
    if each['is_asking']:
        this_pmp += "\n" + inducer_ask.replace("[|__text__|]", last_doc)
    else:
        this_pmp += "\n" + inducer_tell.replace("[|__text__|]", last_doc)
    reasoning = ""
    cnt = 0
    while True:
        if cnt >= 5:
            # find the part of "reasoning": ".*?"
            doc_reasonings.append("")
            break
        try:
            resp = llm.invoke(this_pmp)
            full = this_pmp + resp.content
            full = full.removeprefix(no_inducer)
            start = "assistant\n<tool_call>"
            end = "</tool_call>"
            # find the last start and end position, and take the substring between them
            start_pos = full.rfind(start)
            end_pos = full.rfind(end)
            parsed = json.loads(full[start_pos + len(start):end_pos])
            reasoning = parsed['arguments']['reasoning']
            doc_reasonings.append(reasoning)
            break
        except Exception as e:
            print(e)
            print(full)
            cnt += 1
            continue


# In[15]:


doc_reasonings


# In[18]:


json.dump(doc_reasonings, open("./all_rs.json", "w"), indent=2, ensure_ascii=False)

