.PHONY: all

all: run

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: bulid/Makefile
	cd build && make && ./flowParallel

install:
	cp build/flowParallel ./

clean:
	rm -rf build
