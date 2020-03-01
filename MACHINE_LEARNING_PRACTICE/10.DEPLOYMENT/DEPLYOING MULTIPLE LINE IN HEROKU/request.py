import requests

url = 'https://mymllearn.herokuapp.com/'
r = requests.post(url,json={'tv':2, 'radio':9, 'newspaper':6},verify=False)
#print(r)
print(r.json())