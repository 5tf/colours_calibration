import requests
url = 'http://127.0.0.1:5000/calibrate_colours'
data = open('image.jpg', 'rb').read()
r = requests.post(url, data=data)
