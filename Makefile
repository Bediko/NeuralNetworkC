CC    = gcc

CFLAGS= -g -c -O3 -funroll-loops -funroll-all-loops -fomit-frame-pointer \
	-finline-functions -march=native -mtune=native -Wall -DBLAS -I/usr/local/cuda-5.0/include

DEPS  = C_TrainMLP.h

OBJ   = main.o C_TrainMLP.o 
LFLAGS=-lblas -lOpenCL

all: main

%.o: %.cxx $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o
	rm -f main