import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from config_reader import config
from aiogram import F
from aiogram.types import Message
import transcribe
import summarize
import os
from pathlib import Path
import gc
import urllib.request
import torch
from argparse import ArgumentParser

def show_progress(block_num, block_size, total_size):
    print(round(block_num * block_size / total_size *100,2), '%', end="\r")

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    help='Выбор устройства для вычислений. cpu/cuda')
parser.add_argument('--ngp', type=int, default=0,
                    help='Количество слоёв ллм, которые будут переданы в гпу. Значение "-1" - все слои.')
args = parser.parse_args()
device = args.device.lower()
ngp = args.ngp
MODEL_PATH = Path("mistral-7b-instruct-v0.2.Q5_K_M.gguf")
if not os.path.exists(MODEL_PATH):
    print('Скачивание модели:')
    urllib.request.urlretrieve('https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
                               MODEL_PATH, show_progress)
bot = Bot(token=config.bot_token.get_secret_value(), parse_mode="HTML")
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

@dp.message(Command("start"))
async def send_welcome(message: Message):
    await message.reply("Привет! Я бот, который принимает аудиосообщения и возвращает краткое изложение.")


@dp.message(F.content_type == "voice")
async def handle_voice(message: Message, bot: Bot):

    audio_type = message.voice.mime_type.split('/')[-1]
    audio_path = f"{message.voice.file_id}.{audio_type}"
    await bot.download(
        message.voice,
        destination=audio_path
    )
    await message.reply('Сообщение в обработке.')

    summary = transcribe.transcribe(audio_path, device)
    torch.cuda.empty_cache()
    gc.collect()

    if os.path.exists(audio_path):
        os.remove(audio_path)
    else:
        print("The file does not exist")
    if len(summary.split(' ')) > 75:
        summary = summarize.summarize(MODEL_PATH, summary, n_gpu_layers=ngp)
    torch.cuda.empty_cache()
    gc.collect()

    if summary:
        await message.reply(summary)
    else:
        await message.reply("Запись не содержит речи.")

@dp.message(F.content_type == "audio")
async def handle_audio(message: Message, bot: Bot):
    audio_type = message.audio.mime_type.split('/')[-1]
    audio_path = f"{message.audio.file_id}.{audio_type}"
    await bot.download(
        message.audio,
        destination=audio_path
    )
    await message.reply('Сообщение в обработке.')

    summary = transcribe.transcribe(audio_path, device)
    torch.cuda.empty_cache()
    gc.collect()

    if os.path.exists(audio_path):
        os.remove(audio_path)
    else:
        print("The file does not exist")
    if len(summary.split(' ')) > 75:
        summary = summarize.summarize(MODEL_PATH, summary, n_gpu_layers=ngp)
    torch.cuda.empty_cache()
    gc.collect()

    if summary:
        await message.reply(summary)
    else:
        await message.reply("Запись не содержит речи.")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())