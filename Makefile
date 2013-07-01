CC    = gcc

CFLAGS= -c -O3 -funroll-loops -funroll-all-loops -fomit-frame-pointer \
	-finline-functions -march=native -mtune=native -Wall -DBLAS

DEPS  = C_TrainMLP.h

OBJ   = main.o C_TrainMLP.o 
LFLAGS=-lblas -lOpenCL

%.o: %.cxx $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	g++ -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o
	rm -f main