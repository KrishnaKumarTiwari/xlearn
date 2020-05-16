#!/usr/bin/env python
# coding: utf-8

# # Aviso Data Science Problem Solving Challenge

# ## Problem #3 Coding logic Challenge 
# 
# ### Number name

# Program should accept a number name less than 10000 and print the number
# 
# Ex: One ---> 1
# Ten ----> 10
# twenty two ----> 22
# three hundred and seventy four ----> 374
# eight thousand three hundred and thirty ----> 8630
# 
# b. Submission
# ○ Python Program File (.py) which can be executed.
# 
# c. Evaluation criteria
# ○ Elegance of the programming logic. Handling of error cases, modularity of
# code, adherence to python programming standards etc.

# In[1]:


import sys


# In[2]:


def _input(message,in_type=str):
    while True:
        try:
            return in_type (input(message))
        except:pass


# #### We can restrict user input as integer using _input
#_input("Only accepting integer : ",int)
# In[3]:


# collection of single digits number represenation
single_digits_number = ["zero", "one", "two", "three", "four", "five", 
                        "six", "seven","eight", "nine"]; 


# In[4]:


# collection of double digits number represenation (10-19)
special_two_digits_number = ["", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
              "sixteen", "seventeen", "eighteen","nineteen"]; 


# In[5]:


# collection of 10's multiple
tens_multiple = ["", "", "twenty", "thirty", "forty","fifty", "sixty", "seventy", "eighty","ninety"]; 


# In[6]:


# collection of 100's multiple
hundred_multiple = ["hundred", "thousand"];


# ### Note keeping the range till thousand as ten thousand it out of scope of this question

# In[7]:


# A function that prints given number in words 

def display_word_represenation(num):
    
    try: 
        if(num is not None):
            length = len(num); 
            if (length == 0): 
                print("Not a Valid Input | Invalid Number"); 
                return; 
            if (num.isnumeric()):
                ##check if non negative number
                if (0 > int(num)):
                    print("Not a Valid Input | Negative numbers not supported")
                    return;
            else: 
                print("Not a Valid Input| Input value is not a Number")
                return;
            if (length > 4): 
                print("Limitation | Number greater than 10000 not supported"); 
                return; 
        else:
            print("Please Input a Valid Input | None is not accetable")


        if (length == 1): 
            #ASCII value based position find 
            print(single_digits_number[ord(num[0]) - ord('0')]); 
            return; 

        pointer = 0; 
        while (pointer < len(num)):
            if (length >= 3) :
                if (ord(num[pointer]) - ord('0') != 0):
                    print(single_digits_number[ord(num[pointer]) - ord('0')], end = " "); 
                    print(hundred_multiple[length - 3], end = " ");
                    length -= 1;
            else : 
                if (ord(num[pointer]) - ord('0') == 1): 
                    sum = (ord(num[pointer]) - ord('0') + ord(num[pointer +1]) - ord('0')); 
                    print(special_two_digits_number[sum]);
                    return; 
                elif (ord(num[pointer]) - ord('0') == 2 and ord(num[pointer + 1]) - ord('0') == 0): 
                    print("twenty"); 
                    return;
                else: 
                    i = ord(num[pointer]) - ord('0'); 
                    if(i > 0): 
                        print(tens_multiple[i], end = " "); 
                    else: 
                        print("", end = ""); 
                    pointer += 1; 
                    if(ord(num[pointer]) - ord('0') != 0): 
                        print(single_digits_number[ord(num[pointer]) - 48]); 
            pointer += 1;
    except Exception as e:
        print("Something went wrong") 
        print(e)


# In[8]:


## We can use unittest to define the test cases 

def test():    
    display_word_represenation("-1")
    display_word_represenation("10000")
    display_word_represenation("10.5")
    display_word_represenation("-105")
    display_word_represenation("/u005")
    

#if __name__ == "__main__":
#    test()


# In[9]:


def main(argv):
    display_word_represenation(argv[0])


# In[10]:


if __name__ == "__main__":
   main(sys.argv[1:])

