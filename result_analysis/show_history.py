import json
import fire
import os
from pygments import highlight
from pygments.lexers import JavaLexer
from pygments.formatters import TerminalFormatter

temp_path = '/tmp/chatunitest-info'

data = json.load(open('/Users/chenyi/Documents/sag/Final_Project/data/coverage/hits/mike/experiment/tmp/binance-connector-java/binance-connector-java/history2024_12_23_03_05_59/class39/method5/attempt0/records.json', 'r'))

def show_history(
        project_name: str,
        class_name: str,
        method_name: str,
        tmp_output_path: str = temp_path
):
    project_path = os.path.join(tmp_output_path, project_name)

    method_history_paths = locate_history_path(project_path, class_name, method_name)

    ans = input("Do you want to show the record? (y/n): ")
    if ans.lower() == "y":
        for i, m_path in enumerate(method_history_paths):
            attemp_map = json.load(open(os.path.join(m_path, "attemptMapping.json"), "r"))
            for attempt_index in attemp_map:
                try:
                    record = json.load(open(os.path.join(m_path, attempt_index, "records.json"), "r"))
                    print_record(record)
                    save_records_to_txt(record, f"./history-{i}-{method_name}")
                except IOError:
                    pass



def locate_history_path(
        project_tmp_path: str,
        class_name: str,
        method_name: str
):
    history_dirs = [d for d in os.listdir(project_tmp_path) if os.path.isdir(os.path.join(project_tmp_path, d)) and d.startswith("history")]

    # the project_tmp_path may has multiple history directory named history_*, if not one, show all history and let user to choose one
    assert len(history_dirs) > 0, f"No history directory found in {project_tmp_path}"
    if len(history_dirs) > 1:
        print("History directories:")
        for i, d in enumerate(history_dirs):
            print(f"{i}: {d}")
        history_index = int(input("Please input the index of the history directory: "))
        history_path = os.path.join(project_tmp_path, history_dirs[history_index])
    else:
        history_path = os.path.join(project_tmp_path, history_dirs[0])

    # locate record
    class_map = json.load(open(os.path.join(project_tmp_path, "classMapping.json"), "r"))
    candidate_class = []
    for class_index in class_map:
        if class_map[class_index]["className"] == class_name:
            candidate_class.append(class_index)
    assert len(candidate_class) > 0, f"No class named {class_name} found in {history_path}"

    method_history_paths = []
    for class_index in candidate_class:
        class_history_path = os.path.join(history_path, class_index)
        method_map = json.load(open(os.path.join(class_history_path, "methodMapping.json"), "r"))
        for method_index in method_map:
            if method_map[method_index]["methodName"] == method_name:
                method_history_paths.append(os.path.join(class_history_path, method_index))
    assert len(method_history_paths) > 0, f"No method named {method_name} found in {history_path}"

    print("Method history paths:")
    for i, p in enumerate(method_history_paths):
        print(f"{i}: {p}")

    return method_history_paths


def print_record(record):

    for item in record:
        print("\n" + "#" * 100 + "\n")
        print(f"Attempt: {item['attempt']}, Round: {item['round']}")
        print("\n" + "#" * 100 + "\n")
        for prompt_item in item['prompt']:
            print(f"[{prompt_item['role'].title()}] :\n{prompt_item['content']}\n")
            print("\n" + "=" * 100 + "\n")

        print(f"[Response]:\n{item['response']}")

        if item['hasCode']:
            # 使用Pygments对代码进行格式化和高亮
            formatted_code = highlight(item['code'], JavaLexer(), TerminalFormatter())
            print("Formatted Code:")
            print(formatted_code)
        else:
            print("Response:")
            print(item['response'])

def save_records_to_txt(record, output_dir):
    for item in record:
        filename = f"attempt_{item['attempt']}_round_{item['round']}.txt"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(filepath, 'w') as file:
            file.write("\n" + "#" * 100 + "\n")
            file.write(f"Attempt: {item['attempt']}, Round: {item['round']}\n")
            file.write("\n" + "#" * 100 + "\n")
            for prompt_item in item['prompt']:
                file.write(f"[{prompt_item['role'].title()}] :\n{prompt_item['content']}\n")
                file.write("\n" + "=" * 100 + "\n")

            file.write(f"[Response]:\n{item['response']}\n")

            if item['hasCode']:
                file.write("Formatted Code:\n")
                file.write(item['code'])
            else:
                file.write("Response:\n")
                file.write(item['response'])


if __name__ == "__main__":
    print_record(data)
    # fire.Fire(show_history)