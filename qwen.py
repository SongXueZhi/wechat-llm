from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from collections import defaultdict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime
import json
import os
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent

from langchain_core.documents import Document


class VectorStoreMemory:
    def __init__(self, db_path='chroma_db'):
        self.db_path = db_path
        self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
        self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_model)

    def save_context(self, user_id, message):
        self.vectorstore.add_documents([Document(page_content=message.type+": "+message.content)], collection_name=user_id)
        self.vectorstore.persist()

    def load_context(self, user_id):
        results = self.vectorstore.similarity_search(user_id, top_k=4, collection_name=user_id)
        return [(result.metadata["date_time"], result.page_content) for result in results]

class ChatBot:
    def __init__(self, api_key, model): 
        self.chat_memory = defaultdict(lambda:ChatMessageHistory())
        self.vector_store_memory = VectorStoreMemory()
        self.chat = ChatTongyi(
            api_key=api_key,
            model=model,
        )
        self.vector_store_memory = VectorStoreMemory()

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个聊天机器人，你可以回答任何问题，你很爱交流，请问答！",
                ),
                ("user", "{messages}"),
                ("placeholder", "{contexts}"),
            ]
        )
        self.chain = self.prompt | self.chat

    
    def get_memory(self, chat_name):
        chat_memory = self.chat_memory[chat_name]
        
        if len(chat_memory.messages) > 6:
            out_message = chat_memory.messages.pop(0)
            self.vector_store_memory.save_context(chat_name,out_message)
        return self.chat_memory[chat_name]
    
    def generate_response(self, chat_name, content):
        chat_memory = self.get_memory(chat_name)
        chat_memory.add_user_message(content)
        
        tool = create_retriever_tool(
        self.vector_store_memory.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
        )
        tools = [tool]
        agent_executor = create_react_agent(self.chat, tools)
        long_his = agent_executor.invoke({"messages":[chat_memory.messages[-1]]},)
        response = self.chain.invoke({
            "messages": chat_memory.messages,
            "contexts": long_his["messages"],
        })
        chat_memory.add_ai_message(response)
        return response
        
  
if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = "qwen-turbo"
    bot = ChatBot(api_key=api_key, model=model)
    response = bot.generate_response("1test", "你好，我是宋学志")
    print(response)
    response = bot.generate_response("1test", "我是谁？")
    print(response)