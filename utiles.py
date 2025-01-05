def load_data(path):
    with open(path,'r',encoding='utf-8') as f:
        data=[[line.strip()] for line in f.readlines()]
        return data

def save_data(datax, path):
    with open(path, 'w', encoding="UTF8") as f:
        for lines in datax:
            for i, line in enumerate(lines[0]):
                f.write(str(line))
                if i != len(lines) - 1:
                    f.write(',')
            f.write('\n')