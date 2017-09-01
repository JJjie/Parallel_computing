.PHONY: all

all: run install

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: build/Makefile
	cd build && make && ./performance && cd ../

install:
	cp build/performance ./

clean:
	rm -rf build
	rm performance
