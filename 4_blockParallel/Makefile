.PHONY: all

all: run

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: bulid/Makefile
	cd build && make && ./blockParallel && cd ../

install:
	cp build/blockParallel ./

clean:
	rm -rf build
