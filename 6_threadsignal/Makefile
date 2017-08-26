.PHONY: all

all: run

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: build/Makefile
	cd build && make && ./flowParallel && cd ../

install:
	cp build/flowParallel ./

clean:
	rm -rf build
