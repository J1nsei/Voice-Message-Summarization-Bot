import guidance
from guidance import models, gen, system, assistant, user
from pathlib import Path

def summarize(model_path: Path, text: str, n_ctx: int = 32768, n_gpu_layers: int = 0):
    llm = models.LlamaCppChat(model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False, echo=False)
    with system():
        llm += """Ты - Мистраль, русскоязычный ассистент суммаризации текстов и диалогов. 
                Ты помогаешь пользователям обрабатывать текста сообщений других пользователей. 
                В своём ответе используй только натуральный русский язык.
               """
    with user():
        llm += "Суммаризируй сообщение: " + text
    with assistant():
        llm += gen("response")
    summary = llm['response']
    llm = None
    return summary