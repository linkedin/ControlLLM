
import os
import json


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs


def add_chats(prompt_file: str="./chats.json", system_content: str="", user_content: str=""):
    assert os.path.exists(
        prompt_file
    ), f"Provided Prompt file does not exist {prompt_file}"

    dialogs= read_dialogs_from_file(prompt_file)
    print(f"Read {len(dialogs)} dialogs from {prompt_file}...")
    print("\n==================================\n")

    dialog = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    # insert dialog into dialogs as the second
    dialogs.insert(1, dialog)
    print(f"Inserted a new dialog {dialog} into the dialogs list...")
    print("\n==================================\n")
  
    with open(prompt_file, 'w') as file:
        json.dump(dialogs, file)
    print(f"Saved the dialogs back to {prompt_file}...")

# read from "./chats_initial.json" and write the content to "./chats.json"
chats_initial_file = "./chats_initial.json"
chats_file = "./chats.json"
# read from chats_initial_file
with open(chats_initial_file, 'r') as file:
    chats_initial = json.load(file)
    print(f"Read number of dialogs from {chats_initial_file}: {len(chats_initial)}...")
    # write to chats_file
    with open(chats_file, 'w') as file:
        json.dump(chats_initial, file)
    print(f"Saved the chats from {chats_initial_file} to {chats_file}...")


# New test case

system_content = """
# Identity
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
# Instruction
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

user_content = "How to travel from LA to SF?"

add_chats(prompt_file="./chats.json", system_content=system_content, user_content=user_content)
