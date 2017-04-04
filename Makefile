#
#  Author: Kamil Rocki <kmrocki@us.ibm.com>
#  Created on: 02/23/2016
#

OS := $(shell uname)
CC=g++-6
USE_BLAS=1
INCLUDES=-I.
LFLAGS=
CFLAGS=-Ofast -std=c++11 -fopenmp

ifeq ($(OS),Linux)
	CFLAGS := -mavx2 $(CFLAGS)
	INCLUDES := -I/usr/include/eigen3 $(INCLUDES) 
else
	#OSX
#	CC=g++-5.2.0
	INCLUDES := -I/usr/local/include/eigen3 $(INCLUDES)
endif

ifeq ($(USE_BLAS),1)

	ifeq ($(OS),Linux)
		INCLUDES := -I/opt/OpenBLAS/include $(INCLUDES)
		LFLAGS := -lopenblas -L/opt/OpenBLAS/lib $(LFLAGS)
	else
		#OSX
		INCLUDES := -I/usr/local/opt/openblas/include $(INCLUDES)
		LFLAGS := -lopenblas -L/usr/local/opt/openblas/lib $(LFLAGS)
	endif

	CFLAGS := -DUSE_BLAS $(CFLAGS)

endif

all:
	$(CC) ./conv_mnist.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o conv_mnist
	$(CC) ./conv.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o conv
	$(CC) ./c100.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o c100
	$(CC) ./mlp.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o mlp
