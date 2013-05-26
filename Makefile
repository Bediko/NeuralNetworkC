CC=gcc
CFLAGS=-I.
DEPS = C_TrainMLP.h
OBJ = main.o C_TrainMLP.o 
LFLAGS=
%.o: %.cxx $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)