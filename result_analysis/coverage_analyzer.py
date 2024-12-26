import json
import csv
import os
import sys
import xml.etree.ElementTree as ET
import re
from typing import List, Dict
from colorama import init, Fore

init(autoreset=True)


def parse_method_descriptor(descriptor: str) -> str:
    """
    Parses the method descriptor from the 'desc' attribute in the XML file and returns a string representation of the arguments.
    """
    def get_type_name(descriptor: str) -> str:
        type_mapping = {
            'B': 'byte', 'C': 'char', 'D': 'double', 'F': 'float',
            'I': 'int', 'J': 'long', 'S': 'short', 'Z': 'boolean',
            'V': 'void'
        }
        array_suffix = ''
        while descriptor.startswith('['):
            array_suffix += '[]'
            descriptor = descriptor[1:]
        if descriptor.startswith('L'):
            end_index = descriptor.find(';')
            class_name = descriptor[1:end_index].replace('/', '.')
            short_name = class_name.split('.')[-1]
            return short_name + array_suffix
        return type_mapping.get(descriptor, descriptor) + array_suffix

    param_pattern = re.compile(r'\((.*?)\)')
    match = param_pattern.search(descriptor)
    if not match:
        return "()"

    params = match.group(1)
    param_list = []
    i = 0
    while i < len(params):
        if params[i] in 'BCDFIJSZ':
            param_list.append(get_type_name(params[i]))
            i += 1
        elif params[i] == 'L':
            end_index = params.find(';', i)
            param_list.append(get_type_name(params[i:end_index + 1]))
            i = end_index + 1
        elif params[i] == '[':
            array_type_start = i
            while params[i] == '[':
                i += 1
            if params[i] == 'L':
                end_index = params.find(';', i)
                param_list.append(get_type_name(params[array_type_start:end_index + 1]))
                i = end_index + 1
            else:
                param_list.append(get_type_name(params[array_type_start:i + 1]))
                i += 1

    return f"({', '.join(param_list)})"


def parse_jacoco_report(xml_path, methods_dict):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total_lines_covered = 0
    total_lines_missed = 0
    total_branches_covered = 0
    total_branches_missed = 0
    methods_generated = 0  # 计数器，成功生成的方法数量
    total_methods = sum(len(methods) for methods in methods_dict.values())  # 统计tasklist中的方法总数
    incomplete_methods = []  # 保存覆盖率未达到100%的方法

    for package in root.findall('package'):
        for class_elem in package.findall('class'):
            class_name = class_elem.get('name').replace('/', '.')
            if class_name in methods_dict:
                method_names = methods_dict[class_name]
                for method in class_elem.findall('method'):
                    method_name = method.get('name')
                    method_desc = method.get('desc')
                    parsed_method_desc = parse_method_descriptor(method_desc)
                    full_method_name = f"{method_name}{parsed_method_desc}"

                    if any(m == full_method_name for m in method_names):
                        line_covered = 0
                        line_missed = 0
                        branch_covered = 0
                        branch_missed = 0

                        for counter in method.findall('counter'):
                            counter_type = counter.get('type')
                            missed = int(counter.get('missed'))
                            covered = int(counter.get('covered'))

                            if counter_type == 'LINE':
                                line_missed = missed
                                line_covered = covered
                            elif counter_type == 'BRANCH':
                                branch_missed += missed
                                branch_covered += covered

                        total_lines_missed += line_missed
                        total_lines_covered += line_covered
                        total_branches_missed += branch_missed
                        total_branches_covered += branch_covered

                        # 判断行覆盖率不为0的方法
                        if line_covered > 0:
                            methods_generated += 1
                        if line_missed > 0:
                            incomplete_methods.append(
                                f"{class_name}#{method_name}: {line_covered}/{line_covered + line_missed}"
                            )

    return {
        "total_lines_covered": total_lines_covered,
        "total_lines_missed": total_lines_missed,
        "total_branches_covered": total_branches_covered,
        "total_branches_missed": total_branches_missed,
        "methods_generated": methods_generated,
        "total_methods": total_methods,
        "incomplete_methods": incomplete_methods  # 未完全覆盖的方法列表
    }


def calculate_coverage(coverage_data):
    total_lines = coverage_data["total_lines_covered"] + coverage_data["total_lines_missed"]
    total_branches = coverage_data["total_branches_covered"] + coverage_data["total_branches_missed"]

    line_coverage = coverage_data["total_lines_covered"] / total_lines * 100 if total_lines > 0 else 0
    branch_coverage = coverage_data["total_branches_covered"] / total_branches * 100 if total_branches > 0 else 0
    generation_rate = coverage_data["methods_generated"] / coverage_data["total_methods"] * 100 if coverage_data["total_methods"] > 0 else 0

    return {
        "line_coverage": line_coverage,
        "branch_coverage": branch_coverage,
        "total_lines_covered": coverage_data["total_lines_covered"],
        "total_lines_missed": coverage_data["total_lines_missed"],
        "total_branches_covered": coverage_data["total_branches_covered"],
        "total_branches_missed": coverage_data["total_branches_missed"],
        "total_lines": total_lines,
        "total_branches": total_branches,
        "methods_generated": coverage_data["methods_generated"],
        "total_methods": coverage_data["total_methods"],
        "generation_rate": generation_rate,
        "incomplete_methods": coverage_data["incomplete_methods"]  # 返回未完全覆盖的方法列表
    }


def aggregate_coverage_data(coverage_data_list: List[Dict[str, int]]) -> Dict[str, float]:
    total_lines_covered = sum(data["total_lines_covered"] for data in coverage_data_list)
    total_lines_missed = sum(data["total_lines_missed"] for data in coverage_data_list)
    total_branches_covered = sum(data["total_branches_covered"] for data in coverage_data_list)
    total_branches_missed = sum(data["total_branches_missed"] for data in coverage_data_list)
    total_methods_generated = sum(data["methods_generated"] for data in coverage_data_list)
    total_methods = sum(data["total_methods"] for data in coverage_data_list)
    incomplete_methods = [method for data in coverage_data_list for method in data["incomplete_methods"]]  # 汇总未完全覆盖的方法

    aggregated_data = {
        "total_lines_covered": total_lines_covered,
        "total_lines_missed": total_lines_missed,
        "total_branches_covered": total_branches_covered,
        "total_branches_missed": total_branches_missed,
        "methods_generated": total_methods_generated,
        "total_methods": total_methods,
        "incomplete_methods": incomplete_methods  # 返回未完全覆盖的方法列表
    }

    return calculate_coverage(aggregated_data)


def find_xml(jacocoResultPath, module_name):
    
    for root, dirs, files in os.walk(os.path.join(jacocoResultPath, module_name)):
        for file in files:
            if file == 'jacoco.xml':
                return os.path.join(root, file)
    return None


def muti_module_extract(tmpOutputPath, jacocoResultPath):
    modules = [d for d in os.listdir(tmpOutputPath) if os.path.isdir(os.path.join(tmpOutputPath, d))]
    coverage_data_list = []

    for module in modules:
        module_json_path = os.path.join(tmpOutputPath, module, 'focal_methods_sampled.csv')
        # module_xml_path = os.path.join(jacocoResultPath, module, 'target', 'site', 'jacoco', 'jacoco.xml')
        module_xml_path = find_xml(jacocoResultPath, module)

        if os.path.exists(module_json_path) and os.path.exists(module_xml_path):
            if module_json_path.endswith('.csv'):
                methods_dict = process_csv_to_dict(module_json_path)
            else:
                with open(module_json_path, 'r') as json_file:
                    methods_dict = json.load(json_file)
            # with open(module_json_path, 'r') as json_file:
            #     methods_dict = json.load(json_file)

            module_coverage_data = parse_jacoco_report(module_xml_path, methods_dict)
            coverage_data_list.append(module_coverage_data)

    aggregated_coverage_info = aggregate_coverage_data(coverage_data_list)

    # 打印汇总的覆盖率信息
    print(Fore.GREEN + f"Total Lines: {aggregated_coverage_info['total_lines']}")
    print(Fore.GREEN + f"Total Lines Covered: {aggregated_coverage_info['total_lines_covered']}")
    print(Fore.GREEN + f"Line Coverage: {aggregated_coverage_info['line_coverage']:.2f}%\n")
    print(Fore.YELLOW + f"Total Branches: {aggregated_coverage_info['total_branches']}")
    print(Fore.YELLOW + f"Total Branches Covered: {aggregated_coverage_info['total_branches_covered']}")
    print(Fore.YELLOW + f"Branch Coverage: {aggregated_coverage_info['branch_coverage']:.2f}%\n")
    print(Fore.CYAN + f"Methods Generated: {aggregated_coverage_info['methods_generated']}")
    print(Fore.CYAN + f"Total Methods: {aggregated_coverage_info['total_methods']}")
    print(Fore.CYAN + f"Generation Rate: {aggregated_coverage_info['generation_rate']:.2f}%\n")


def single_module_extract(jacoco_xml_path, json_path, incomplete_covered_method_json_path=None):

    if json_path.endswith('.csv'):
        methods_dict = process_csv_to_dict(json_path)
    else:
        with open(json_path, 'r') as json_file:
            methods_dict = json.load(json_file)

    coverage_data = parse_jacoco_report(jacoco_xml_path, methods_dict)
    coverage_info = calculate_coverage(coverage_data)

    # 打印单个模块的覆盖率信息
    print(Fore.GREEN + f"Total Lines: {coverage_info['total_lines']}")
    print(Fore.GREEN + f"Total Lines Covered: {coverage_data['total_lines_covered']}")
    print(Fore.GREEN + f"Line Coverage: {coverage_info['line_coverage']:.2f}%\n")
    print(Fore.YELLOW + f"Total Branches: {coverage_info['total_branches']}")
    print(Fore.YELLOW + f"Total Branches Covered: {coverage_data['total_branches_covered']}")
    print(Fore.YELLOW + f"Branch Coverage: {coverage_info['branch_coverage']:.2f}%\n")
    print(Fore.CYAN + f"Methods Generated: {coverage_info['methods_generated']}")
    print(Fore.CYAN + f"Total Methods: {coverage_info['total_methods']}")
    print(Fore.CYAN + f"Generation Rate: {coverage_info['generation_rate']:.2f}%\n")

    # 保存未完全覆盖的方法到指定路径的 JSON 文件
    if incomplete_covered_method_json_path:
        with open(incomplete_covered_method_json_path, 'w') as outfile:
            json.dump(coverage_info['incomplete_methods'], outfile, indent=4)

def process_csv_to_dict(csv_file):
    # 创建一个字典来存储每个类和其方法
    class_methods = {}

    # 读取 CSV 文件
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # 跳过表头
        next(reader)
        
        # 遍历每一行数据
        for row in reader:
            class_name = row[0]
            method_signature = row[1]
            
            # 将方法添加到对应类的列表中
            if class_name in class_methods:
                class_methods[class_name].append(method_signature)
            else:
                class_methods[class_name] = [method_signature]
    
    return class_methods


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Please provide the mode and the required arguments.")
    else:
        mode = sys.argv[1]
        jacoco_xml_path = sys.argv[2]
        tasklist_json_path = sys.argv[3]
        
    if mode == "multi":
        muti_module_extract(tasklist_json_path, jacoco_xml_path)
    elif mode == 'single':
        incomplete_covered_method_record_file = input("output_json_path (leave empty if not saving):")
        output_json_path = incomplete_covered_method_record_file if incomplete_covered_method_record_file.strip() else None
        if output_json_path:
            output_json_path = os.path.join(output_json_path, 'incomplete_covered_method_record.json')
        single_module_extract(jacoco_xml_path, tasklist_json_path, output_json_path)
