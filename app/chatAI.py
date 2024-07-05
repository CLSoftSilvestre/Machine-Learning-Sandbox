from datetime import datetime
import ollama


class ChatAI():

    chatHistory = []
    
    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    def AskQuestion(self, strQuestion):
        self.chatHistory.append(ChatMessage("User",strQuestion))

        response = ollama.chat(model = self.model, messages=[
            {
                'role' : 'user',
                'content' : strQuestion,
                'stream' : False,
            },
        ])

        self.chatHistory.append(ChatMessage("Model", response['message']['content']))

        return response['message']['content']
      
    def ClearHistory(self):
        self.chatHistory = []


class ChatMessage():

    def __init__(self, role, message):
        self.role = role
        self.message = message
        self.timestamp = datetime.now

