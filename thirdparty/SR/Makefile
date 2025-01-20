CXX=gcc
FLAGS=-Wall -O3 -march=native -std=c11 -lm
BIN=SR

FLAGS+=-Wno-unknown-pragmas

all: $(BIN)

SR: SR_HEAP.c
	${CXX} SR_HEAP.c ${FLAGS} -o ${BIN}

clean:
	rm -f $(BIN) *.o


