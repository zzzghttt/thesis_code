import os
import json
from posix import error
import fire

def use_mock(content: str) -> bool:
    if content is None:
        return False
    if "@Mock" in content or "when(" in content:
        return True
    return False

def has_mock_error(content: str) -> bool:
    if content is None:
        return False
    if "org.mockito.exceptions.verification" in content or "org.mockito.exceptions.misusing" in content or "org.mockito.exceptions.base.MockitoException" in content or "org.mockito.exceptions" in content:
        return True
    return False

def analyze_records(history_path):
    prompt_token = 0
    response_token = 0
    generation_error_count = 0
    generation_total_count = 0
    repair_error_count = 0
    repair_total_count = 0
    mock_error_count = 0
    mock_total_count = 0
    result_dict = {
        "request_count": 0,        
        "extract_error_count": 0,
        "compile_error_count": 0,
        "runtime_error_count": 0,
        "unknown_error_count": 0,
    }

    # 假设 walk_over_find_this 类似于 os.walk，用于遍历目录
    for root, dirs, files in os.walk(history_path):
        for file in files:
            if file == "records.json":
                record_path = os.path.join(root, file)
                try:
                    with open(record_path, 'r') as f:
                        record_dict = json.load(f)
                except FileNotFoundError:
                    print(f"Record file not found: {record_path}")
                    raise

                for round_dict in record_dict:     
                    if round_dict.get('round') == 0:
                        generation_total_count += 1
                    else:
                        repair_total_count += 1

                    if use_mock(round_dict.get("code")):
                        mock_total_count += 1

                    result_dict["request_count"] += 1
                    prompt_token += int(round_dict.get("promptToken"))
                    response_token += int(round_dict.get("responseToken"))
                    if round_dict.get("hasCode") == False:
                        if round_dict.get('round') == 0:
                            generation_error_count += 1
                        else:
                            repair_error_count += 1
                        result_dict["extract_error_count"] += 1
                    elif round_dict.get("hasError") == True:
                        if round_dict.get('round') == 0:
                            generation_error_count += 1
                        else:
                            repair_error_count += 1

                        error_msg = round_dict.get("errorMsg")
                        if error_msg is None:
                            result_dict["unknown_error_count"] += 1
                        else:
                            if error_msg.get("errorType") == "COMPILE_ERROR":
                                result_dict["compile_error_count"] += 1
                            elif error_msg.get("errorType") == "RUNTIME_ERROR":
                                result_dict["runtime_error_count"] += 1
                            else:
                                print(f"Unknown error type: {error_msg.get('errorType')}")
                                raise
                                

                            error_msg_contents = error_msg.get("errorMessage")
                            for line in error_msg_contents:
                                if has_mock_error(line):
                                    mock_error_count += 1

    print("Record Path: " + history_path)
    print("Analysis result:")
    print(f"Prompt Token: {prompt_token}")
    print(f"Response Token: {response_token}")
    print(f"Generation Error Rate: {generation_error_count / generation_total_count * 100:.2f}%")
    print(f"Repair Error Rate: {repair_error_count / repair_total_count * 100:.2f}%")
    
    print(f"Mock Error Count: {mock_error_count}")
    print(f"Mock Total Count: {mock_total_count}")
    if mock_total_count == 0:
        print("Mock Error Rate: 0.00%")
    else:
        print(f"Mock Error Rate: {mock_error_count / mock_total_count * 100:.2f}%")
    

    print("\nError Distribution:")
    for key, value in result_dict.items():
        print(f"{key}: {value} (rate: {value / result_dict['request_count'] * 100:.2f}%)")
    print(f"total error rate: {(result_dict['extract_error_count'] + result_dict['compile_error_count'] + result_dict['runtime_error_count']) / result_dict['request_count'] * 100:.2f}%")


if __name__ == "__main__":
    fire.Fire(analyze_records)
