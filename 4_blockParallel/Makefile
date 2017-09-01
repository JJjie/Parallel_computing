.PHONY: all

all: run install

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: build/Makefile
	cd build && make && ./blockParallel && cd ../

install:
	cp build/blockParallel ./

clean:
	rm -rf build
	rm blockParallel
