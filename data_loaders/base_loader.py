from abc import ABC, abstractmethod
import json

class BaseLoader(ABC):
    # Stop sequences for LLM generation. Override in subclasses if needed.
    stop_sequences = None  # None = use default stop sequences
    # Whether to enable thinking mode (e.g. Qwen3 <think> tags).
    # Default False so model outputs Action/JSON directly.
    enable_thinking = False
    
    def __init__(self, data_path, agent_name=None):
        self.data_path = data_path
        self.agent_name = agent_name  # Used by strategies that need an LLM preprocessor
        self.data = self.load_data()

    @abstractmethod
    def load_data(self):
        """Đọc và trả về danh sách các dict chứa dữ liệu thô."""
        pass

    @abstractmethod
    def format_prompt(self, sample):
        """Format sample thành prompt dạng string để đưa vào LLM."""
        pass
        
    @abstractmethod
    def get_ground_truth(self, sample):
        """Lấy đáp án (ground truth) từ sample để tính metric."""
        pass
        
    @abstractmethod
    def get_question(self, sample):
        """Lấy câu hỏi/task từ sample để đưa vào LLM Judge."""
        pass
