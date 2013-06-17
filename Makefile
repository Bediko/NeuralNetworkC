CC=g++
CFLAGS=-g -Wall -I.
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