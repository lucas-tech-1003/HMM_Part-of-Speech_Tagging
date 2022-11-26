import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup


URL = "https://q.utoronto.ca/courses/278996/pages/part-of-speech-tags-for-assignment-4"

driver = webdriver.Chrome()
driver.get(URL)
time.sleep(5)
username = driver.find_element(By.ID, "username")
password = driver.find_element(By.ID, "password")

username.send_keys('leihuipe')
password.send_keys('')
login = driver.find_element(By.CLASS_NAME, "btn-lg")
login.click()

time.sleep(5)
p_tags = driver.find_elements(By.TAG_NAME, "p")
text = []
for i in range(2, 63):
    p_text = p_tags[i].text.split(" ")[0]
    text.append(p_text)
for i in range(64, len(p_tags)):
    text.extend(p_tags[i].text.split(" "))
print(text)
