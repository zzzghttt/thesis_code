import csv
import fire

def read_coverage_data(csv_file):
    data = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过头部
        for row in reader:
            key = (row[0], row[1], row[2])  # Package, Class, Method
            coverage_data = {
                'line_coverage': row[3],
                'covered_lines': int(row[8]),
                'total_lines': int(row[9])
            }
            data[key] = coverage_data
    return data

def compare_coverages(file1, file2, output_file):
    data1 = read_coverage_data(file1)
    data2 = read_coverage_data(file2)

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Package', 'Class', 'Method', 'Covered Lines Difference', 'Total Lines'])

        for key in data1:
            if key in data2:
                covered_lines_diff = data2[key]['covered_lines'] - data1[key]['covered_lines']
                total_lines = data1[key]['total_lines']
                writer.writerow([key[0], key[1], key[2], covered_lines_diff, total_lines])

if __name__ == "__main__":
    fire.Fire(compare_coverages)
