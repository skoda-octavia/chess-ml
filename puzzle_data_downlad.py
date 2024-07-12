import requests

url = 'https://wtharvey.com/'

filenames = [
    'm8n2.txt',
    'm8n3.txt',
    'm8n4.txt'
]
for local_filename in filenames:
    temp_url = url + local_filename
    response = requests.get(temp_url)
    response.raise_for_status()

    with open(local_filename, 'wb') as file:
        file.write(response.content)

    print(f"Pobrano plik: {local_filename}")