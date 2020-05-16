#!/usr/bin/env python
# coding: utf-8
import sys

def skyline_analyzer(skyline_2d):
    height=[]
    current_max=0
    location_tallest_building =-1
    impossible_constructions = []
    floor_count=0
    found_brick=False
    print(skyline_2d[0])
    for i in range(len(skyline_2d[0])): 
        for j in range(len(skyline_2d)): 
            if(skyline_2d[j][i]==1):
                found_brick=True
                floor_count+=skyline_2d[j][i]
            if(found_brick==True and skyline_2d[j][i]==0) :
                impossible_constructions.append(i+1)
        if (current_max<floor_count) :
            current_max = floor_count
            location_tallest_building = i+1
        floor_count=0
        found_brick=False
    print("Height of the tallest skyscraper", current_max)
    print("Location of the sky scrapper | Tower No ", location_tallest_building)
    print("Impossible constructions | Tower Nos | ",impossible_constructions)



def main(argv):
#     argv[0] = [
#   [1, 1, 0, 1],
#   [1, 1, 1, 1],
#   [1, 1, 0, 0],
#   [1, 1, 1, 1]
# ]
    skyline_analyzer(argv[0])


if __name__ == "__main__":
   main(sys.argv[1:])




