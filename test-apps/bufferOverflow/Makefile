all:
	nvcc -ccbin=$(CXX) -G -O3 -arch=sm_75 bufferOverflow.cu -o bufferOverflow

clean:
	rm -f bufferOverflow
