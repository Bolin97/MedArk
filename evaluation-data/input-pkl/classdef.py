
# LOAD THIS FILE SO THAT YOU CAN LOAD THE PICKLE FILE UNDERT THE SAME FOLDER
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
