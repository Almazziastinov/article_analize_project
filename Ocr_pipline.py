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
