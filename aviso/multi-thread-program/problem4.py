def rearranged_difference(num):
    return int(''.join(sorted(list(str(num)),reverse=True)))-int(''.join(sorted(list(str(num)))))

def main(argv):
    print(rearranged_difference(argv[0]))

import sys
if __name__ == "__main__":
   main(sys.argv[1:])

