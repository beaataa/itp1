"""
This script uses the Google Drive API to rename and transfer files from one folder to another. 
It first authenticates using OAuth 2.0, either loading existing credentials from token.pickle or prompting the user to log in. 
It then lists files in a specified source folder, renames them by adding the specified prefix, and moves them to the target folder. 
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import os.path

def rename_and_transfer_files():
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    try:
        # Get the folder IDs (you'll need to replace these with your actual folder IDs)
        nut2_folder_id = '1afQlMlK7sZR0mQVy3DexCdwswNzG3HNU'
        nut4_folder_id = '11OUUmnb8gSlWEOfeiTrU7i5_6PoKfq6t'

        # List files in nut4 folder
        results = service.files().list(
            q=f"'{nut4_folder_id}' in parents",
            fields="files(id, name)").execute()
        files = results.get('files', [])

        for file in files:
            # Create new name by adding prefix 'nut4_'
            new_name = f"nut4_{file['name']}"
            
            # Update file metadata with new name
            service.files().update(
                fileId=file['id'],
                body={'name': new_name},
                addParents=nut2_folder_id,
                removeParents=nut4_folder_id
            ).execute()
            
            print(f"Renamed and moved: {file['name']} -> {new_name}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    rename_and_transfer_files()
