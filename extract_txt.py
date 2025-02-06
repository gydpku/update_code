import os
import pdb
def get_txt_content(path):
    text_contents = []

    for root, _, files in os.walk(path): 
        for file in files:
            if file.endswith(".txt"):  # Ensure ".txt" is checked correctly
                file_path = os.path.join(root, file)  # Correct path joining
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_contents.append(f.read())  # Use append to store file content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                if text_contents[-1]=='\n':
                    del text_contents[-1] #pdb.set_trace()
    return text_contents
path='./хПгшЕФф┐охдНхнжхЖЕхо╣'
contents=get_txt_content(path)
pdb.set_trace()
