import requests
import time
import random
from bs4 import BeautifulSoup
import cv2
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
class TextExtractor:
    def __init__(self, languages=['hu', 'en']):
        """
        Инициализирует экстрактор текста с указанными языками
        :param languages: список языков для распознавания (по умолчанию венгерский и английский)
        """
        self.reader = easyocr.Reader(languages)

    def extract_text_from_image(self, images):
        """
        Извлекает текст из изображения по указанному пути
        :param image_url: путь к файлу изображения
        :return: распознанный текст, объединенный в одну строку
        """
        texts = []
        for image in images:
            results = self.reader.readtext(image)  # Можно использовать img_cv или img_np
            texts.append(' '.join([res[1] for res in results]))

        return texts

    def parser(self, url, source):
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        urls = [img['src'] for img in soup.find_all('img') if img.get('src')]
        if source == 'kriptoworld':
            image_urls = [url for url in urls if 'image' in url.lower() and not url.startswith('data:image/svg+xml')]
        else:
            image_urls = [url for url in urls if 'googleusercontent.com' in url]
        return image_urls

    def images_installing(self, urls):
        # 1. Скачиваем изображение по URL
        images = []
        for url in urls:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            # 2. Конвертируем в numpy array (формат, который понимает OpenCV/EasyOCR)
            images.append(np.array(img))
        return images

    def extract_text(self, urls, source):
        texts = []
        for url in urls:
            image_urls = self.parser(url, source)
            images = self.images_installing(image_urls)
            texts.append(self.extract_text_from_image(images))
        return texts
