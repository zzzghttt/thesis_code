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

def parse_jacoco_xml_report(
        file_path: str,
        output_path: str = "./method_coverage.csv"
):
    tree = ET.parse(file_path)
    root = tree.getroot()

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入CSV头部
        writer.writerow(['Package', 'Class', 'Method', 'Line Coverage (%)', 'Branch Coverage (%)', 'Method Coverage (%)', 'Instruction Coverage (%)', 'Complexity Coverage (%)', 'Covered Lines', 'Total Lines'])

        for package in root.findall(".//package"):
            package_name = package.get('name').replace('/', '.')
            for class_element in package.findall(".//class"):
                class_name = class_element.get('name').split('/')[-1]
                for method in class_element.findall(".//method"):
                    method_name = method.get('name')
                    line_coverage = get_coverage(method, "LINE")
                    covered_lines, missed_lines = get_coverage_data(method, "LINE")
                    total_lines = covered_lines + missed_lines
                    branch_coverage = get_coverage(method, "BRANCH")
                    method_coverage = get_coverage(method, "METHOD")
                    instruction_coverage = get_coverage(method, "INSTRUCTION")
                    complexity_coverage = get_coverage(method, "COMPLEXITY")
                    writer.writerow([package_name, class_name, method_name, line_coverage, branch_coverage, method_coverage, instruction_coverage, complexity_coverage, covered_lines, total_lines])

if __name__ == "__main__":
    fire.Fire(parse_jacoco_xml_report)