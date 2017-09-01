.PHONY: all

all: run install

build/Makefile: CMakeLists.txt
	mkdir build && cd build && cmake ../ && cd ../

run: build/Makefile
	cd build && make && ./threadsignal && cd ../

install:
	cp build/threadsignal ./

clean:
	rm -rf build
	rm threadsignal
