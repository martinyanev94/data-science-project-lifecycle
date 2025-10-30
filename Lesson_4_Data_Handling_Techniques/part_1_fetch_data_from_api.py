import requests

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error fetching data from URL")

url = "https://api.example.com/data"
data = fetch_data(url)
print(data)
