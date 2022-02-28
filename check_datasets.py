import pickle
import json

dat = pickle.load(open('datasets/carb_test.pkl', 'rb'))

carb_dev = pickle.load(open('datasets/carb_dev.pkl', 'rb'))

openie4_train = pickle.load(open('datasets/openie4_train.pkl', 'rb'))

structured_data = json.load(open('datasets/structured_data.json', 'r'))


for i in range(50, 70):
    print(structured_data[i]['sentence'])