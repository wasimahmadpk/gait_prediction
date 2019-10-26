import requests
import json
import pandas as pd
import pickle
import numpy as np


url = 'http://127.0.0.1:5000/prediction'

with open(r'C:\Users\Waseem\Desktop\Orikami\xtest.pkl', 'rb') as f:
     data = pickle.load(f)
     data = data.to_json(orient='records')
     print(type(data))

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=data, headers=headers)
print(r, r.text)
