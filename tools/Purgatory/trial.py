import sys
import os
head, tail = os.path.split('a.dat')
print(head+tail)
print("This is the name of the script:", sys.argv[0])
print("Number of arguments:", len(sys.argv))
print("The arguments are:" , str(sys.argv))
