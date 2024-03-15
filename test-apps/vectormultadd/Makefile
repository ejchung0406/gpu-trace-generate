all:
	nvcc -ccbin=$(CXX) -G -O3 -arch=sm_75 vectormultadd.cu -o vectormultadd

clean:
	rm -f vectormultadd
