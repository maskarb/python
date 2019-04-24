import ctypes


print(ctypes.CDLL('./library.so').square(4)) # linux or when mingw used on windows