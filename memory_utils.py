from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.tools.retriever import create_retriever_tool

class VectorStoreMemory:
    def __init__(self, db_path='chroma_db'):
        self.db_path = db_path

    def get_vectorstore(self, user_id):
        return Chroma(persist_directory=self.db_path, collection_name=user_id, embedding=OpenAIEmbeddings())

    def save_context(self, user_id, role, content):
        vectorstore = self.get_vectorstore(user_id)
        doc = {"user_id": user_id, "role": role, "content": content}
        vectorstore.add_documents([doc], collection_name=user_id)
        vectorstore.persist()

    def load_context(self, user_id):
        vectorstore = self.get_vectorstore(user_id)
        results = vectorstore.similarity_search(user_id, top_k=1000, collection_name=user_id)
        return [(result.metadata["role"], result.page_content) for result in results]

class CustomChatHistoryManager:
    def __init__(self, max_length=10, memory_storage=None, user_id=None):
        self.chat_history = ChatMessageHistory()
        self.max_length = max_length
        self.memory_storage = memory_storage
        self.user_id = user_id

    def add_message(self, message):
        self.chat_history.add_message(message)
        if len(self.chat_history.messages) > self.max_length:
            # 将最老的记录存储到向量数据库中
            excess_message = self.chat_history.messages.pop(0)
            if self.memory_storage and self.user_id:
                self.memory_storage.save_context(self.user_id, excess_message.role, excess_message.content)
    
def create_retriever_tool_instance(database):
    return create_retriever_tool(
        retriever_tool,
        "history_retriever",
        "Searches and returns excerpts from the user's chat history.",
    )