.PHONY: all

all: run install

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: build/Makefile
	cd build && make && ./flowParallel && cd ../

install:
	cp build/flowParallel ./

clean:
	rm -rf build
	rm flowParallel
