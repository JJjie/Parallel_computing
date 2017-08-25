.PHONY: all

all: run

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: bulid/Makefile
	cd build && make && ./threadParallel

install:
	cp build/threadParallel ./

clean:
	rm -rf build
