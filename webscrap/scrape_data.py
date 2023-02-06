# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:25:34 2023

@author: Akhilesh
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import codecs
import re
from webdriver_manager.chrome import ChromeDriverManager

class ScrapeData:
    
    def __init__(self,  url, path, DRIVER_PATH):
        self.url = url
        self.path = path
        self.DRIVER_PATH = DRIVER_PATH
        chrome_options = webdriver.ChromeOptions()
        prefs = {'download.default_directory' : r'%s'%self.url}
        chrome_options.add_experimental_option('prefs', prefs)
        self.driver = webdriver.Chrome(executable_path=self.DRIVER_PATH, chrome_options=chrome_options)
        self.driver.implicitly_wait(10)
        self.driver.get("%s"%self.path)
        
    def extract_data(self):
        row = self.driver.find_elements(By.XPATH, '//*[@id="primary"]/table/tbody/tr')