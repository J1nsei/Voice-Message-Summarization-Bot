# Voice-Message-Summarization-Bot

## Description
Телеграм бот для краткого изложения содержания голосовых сообщений / аудиозаписей без использования сторонних API и хранения данных вне серверов телеграма.
## Requirements
- Python 3.10
- Минимум 16 гб ОЗУ при использовании только на CPU
- 8 гб ОЗУ и 12 гб видеопамяти при использовании только на GPU (CUDA)
- ОС: Linux / Windows 10, 11
## Setup
1. `pip install -r requirements.txt`
2. `pip install git+https://github.com/m-bain/whisperx.git`
3. Установите [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
4. Создайте файл `.env` и поместите в него телеграм токен бота 
`BOT_TOKEN = 1234567:AfdseRTFsdwq`
## Usage
1. `python bot.py --device=cuda --ngp=-1` для работы только на **GPU**

   `python bot.py --device=cpu --ngp=0` для работы только на **CPU**

    `python bot.py --device=cuda --ngp=15` гибридный режим: транскрибация на **GPU**, часть суммаризирования на **GPU (15 слоёв)**, остальное на **CPU**

    `python bot.py --device=cpu --ngp=5` гибридный режим: транскрибация на **CPU**, часть суммаризирования на **GPU (5 слоёв)**, остальное на **CPU**

2. Откройте своего бота и отправьте ему команду `/start`. Перешлите боту голосовое сообщение.
