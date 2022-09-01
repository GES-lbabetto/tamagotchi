import os

def main():
    dirname = os.path.dirname(__file__)
    os.system(f"streamlit run {dirname}/1_📂_File_Manager.py --server.headless true")
