import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main():
    # Create data/raw directory if it doesn't exist
    raw_data_path = os.path.join('data', 'raw')
    os.makedirs(raw_data_path, exist_ok=True)

    # Google Drive IDs for your files
    train_file_id = '1_MFcOTadsHaIp3CN-l3FDJ-_gwzAaCAF'
    test_file_id = '1SatIy2P1-el3_nbua_a57WscNZ3jKoO9'

    # File paths for the downloaded files
    train_destination = os.path.join(raw_data_path, 'train.csv')
    test_destination = os.path.join(raw_data_path, 'test.csv')

    # Download the files
    print('Downloading train.csv...')
    download_file_from_google_drive(train_file_id, train_destination)
    
    print('Downloading test.csv...')
    download_file_from_google_drive(test_file_id, test_destination)

    print('Files downloaded successfully.')

if __name__ == "__main__":
    main()
