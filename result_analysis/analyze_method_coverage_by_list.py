import os.path
import xml.etree.ElementTree as ET
import fire
import csv

def get_coverage_data(counter, type):
    element = counter.find(f'.//counter[@type="{type}"]')
    if element is not None:
        covered = int(element.get('covered'))
        missed = int(element.get('missed'))
        return covered, missed
    return 0, 0

def get_coverage(counter, type):
    element = counter.find(f'.//counter[@type="{type}"]')
    if element is not None:
        covered = int(element.get('covered'))
        missed = int(element.get('missed'))
        total = covered + missed
        coverage = (covered / total) * 100 if total > 0 else 0
        return f"{coverage:.2f}"
    return "0.00"

def parse_descriptor(descriptor):
    basic_types = {
        'B': 'byte',
        'C': 'char',
        'D': 'double',
        'F': 'float',
        'I': 'int',
        'J': 'long',
        'S': 'short',
        'Z': 'boolean'
    }

    params = []
    array_depth = 0
    i = 1  # 跳过开头的 '('

    while i < len(descriptor):
        if descriptor[i] == ')':
            break  # 参数列表结束

        if descriptor[i] == '[':
            array_depth += 1
            i += 1
            continue

        if descriptor[i] in basic_types:
            # 基本数据类型
            param_type = basic_types[descriptor[i]]
            if array_depth > 0:
                param_type = param_type + '[]' * array_depth
                array_depth = 0
            params.append(param_type)
            i += 1
        elif descriptor[i] == 'L':
            # 对象类型
            semicolon_index = descriptor.find(';', i)
            param_type = descriptor[i + 1:semicolon_index].replace('/', '.')
            if array_depth > 0:
                param_type = param_type + '[]' * array_depth
                array_depth = 0
            param_type = param_type.split('.')[-1]  # 只保留类名
            params.append(param_type)
            i = semicolon_index + 1
        else:
            raise ValueError(f"Unexpected character '{descriptor[i]}' in descriptor")

    return params

def parse_jacoco_xml_report(
        module_name: str,
        xml_path: str,
        methods_csv_path: str,
        output_path: str = "./method_coverage.csv"
):
    method_data = {}
    with open(methods_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过头部
        for row in reader:
            full_class_name = row[0]
            # get package name
            class_name = full_class_name.split('.')[-1]
            package_name = full_class_name.split('.' + class_name)[0]
            method_sig = row[1]
            method_name = method_sig.split('(')[0]
            method_args = method_sig.split('(')[1].split(')')[0].split(',')
            # stip each arg in args
            method_args = [arg.strip() for arg in method_args]
            method_sig = method_name + '(' + ','.join(method_args) + ')'
            method_data[(full_class_name, method_sig)] = [module_name, package_name, class_name, method_sig, "0.00", "0.00", "0.00", "0.00", "0.00", 0, 0, 0, 0]

    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for package in root.findall(".//package"):
            package_name = package.get('name').replace('/', '.')
            for class_element in package.findall(".//class"):
                class_name = class_element.get('name').split('/')[-1]
                if "$" in class_name:
                    class_name = class_name.split("$")[1]
                full_class_name = package_name + '.' + class_name

                for method in class_element.findall(".//method"):
                    method_name = method.get('name')
                    xml_args = parse_descriptor(method.get('desc'))
                    xml_args = [arg.strip() for arg in xml_args]
                    method_sig = method_name + '(' + ','.join(xml_args) + ')'
                    key = (full_class_name, method_sig)

                    if key in method_data:
                        line_coverage = get_coverage(method, "LINE")
                        covered_lines, missed_lines = get_coverage_data(method, "LINE")
                        total_lines = covered_lines + missed_lines
                        branch_coverage = get_coverage(method, "BRANCH")
                        covered_branches, missed_branches = get_coverage_data(method, "BRANCH")
                        total_branches = covered_branches + missed_branches

                        method_coverage = get_coverage(method, "METHOD")
                        instruction_coverage = get_coverage(method, "INSTRUCTION")
                        complexity_coverage = get_coverage(method, "COMPLEXITY")
                        method_data[key] = [module_name, package_name, class_name, method_sig, line_coverage, branch_coverage, method_coverage, instruction_coverage, complexity_coverage, covered_lines, total_lines, covered_branches, total_branches]

    # 步骤 3: 写入最终结果
    with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Module', 'Package', 'Class', 'Method', 'Line Coverage (%)', 'Branch Coverage (%)', 'Method Coverage (%)', 'Instruction Coverage (%)', 'Complexity Coverage (%)', 'Covered Lines', 'Total Lines', 'Covered Branches', 'Total Branches'])
        for data in method_data.values():
            writer.writerow(data)

if __name__ == "__main__":
    # fire.Fire(parse_descriptor('(Lorg/apache/commons/cli/Options;[Ljava/lang/String;)Lorg/apache/commons/cli/CommandLine;'))
    fire.Fire(parse_jacoco_xml_report)