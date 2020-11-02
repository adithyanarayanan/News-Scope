## Reads a text file and outputs it as a single string for the analyzer in main.py to read

def create_string(file):
    filename = open(file, 'r')
    str = ""
    for line in filename:
        # print(line)
        str += line
    return str

