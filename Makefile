CC=gcc
CFLAGS=-g -Wall -I.
DEPS = C_TrainMLP.h
OBJ = main.o C_TrainMLP.o 
LFLAGS=
%.o: %.cxx $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o
	rm -f main