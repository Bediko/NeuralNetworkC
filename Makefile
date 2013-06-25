CC=g++
CFLAGS= -Wall -I. -O3 -march=native -funroll-loops -funroll-all-loops -floop-optimize -finline-functions
LDFLAGS=-lOpenCL
DEPS = C_TrainMLP.h
OBJ = main.o C_TrainMLP.o 
BIN=main

%.o: %.cxx $(DEPS)
	$(CC) $(CFLAGS) $(LDFLAGS) -c $<

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) -o $(BIN)

clean:
	rm -f *.o
	rm -f main