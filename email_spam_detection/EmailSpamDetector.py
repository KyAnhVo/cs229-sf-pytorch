import NB
import sys
import csv

def getchar(fptr):
    pass

def main():
    assert len(sys.argv) == 2 and sys.argv[1].endswith('.csv'), 'Expect CSV file'

    with open(file= sys.argv[1], mode= 'r', encoding= 'utf-8') as file:
        reader = csv.reader(file, delimiter= ',')
        for row in reader:
            print(', '.join(row))
        







if __name__ == '__main__':
    main()
