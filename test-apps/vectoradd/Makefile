all:
	nvcc -ccbin=$(CXX) -G -O3 -arch=sm_75 vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
