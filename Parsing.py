import requests
import time
import random
from bs4 import BeautifulSoup
import cv2
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np

def images_installing(urls):
    # 1. Скачиваем изображение по URL
    images = []
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # 2. Конвертируем в numpy array (формат, который понимает OpenCV/EasyOCR)
        images.append(np.array(img))
    return images

def parcer(url, source):
    delay = random.uniform(0.5, 3.0)  # Случайная задержка от 0.5 до 2 секунд
    time.sleep(delay)
    try:
        # Отправляем GET-запрос
        response = requests.get(url)
        response.raise_for_status()  # Проверяем на ошибки

        # Создаем объект BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = [img['src'] for img in soup.find_all('img') if img.get('src')]

        if source == 'kriptoworld':
        # Находим нужный div
            imgs = images_installing([url for url in urls if 'image' in url.lower() and not url.startswith('data:image/svg+xml')])

            entry_content = soup.find('div', class_='entry-content clearfix single-post-content')

            if entry_content:
                # Очищаем от ненужных элементов
                for div in entry_content.find_all('div', class_='code-block'):
                    div.decompose()

                # Удаляем пустые абзацы
                for p in entry_content.find_all('p'):
                    if not p.get_text(strip=True):
                        p.decompose()

                # Получаем текст с сохранением структуры
                clean_content = entry_content.get_text(separator='\n', strip=True)

                return clean_content, imgs
            else:

                imgs = images_installing([url for url in urls if 'googleusercontent.com' in url])
                entry_content = soup.find('div', class_='td_block_wrap tdb_single_content tdi_85 td-pb-border-top td_block_template_1 td-post-content tagdiv-type')
                content_block = entry_content.find('div', class_='tdb-block-inner td-fix-index')
                if content_block:
                    # Удаляем ненужные элементы (скрипты, стили и т.д.)
                    for element in content_block(['script', 'style', 'a']):
                        element.decompose()

                    # Получаем чистый текст с сохранением структуры
                    clean_text = '\n'.join([p.get_text(strip=True) for p in content_block.find_all(['p', 'h2']) if p.get_text(strip=True)][:-1])

                    return clean_text, imgs


            print("Контент успешно сохранен в файл parsed_content.txt")
        else:
            print("Не удалось найти контент на странице")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к сайту: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
