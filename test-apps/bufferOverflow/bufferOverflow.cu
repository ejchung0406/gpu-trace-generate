#include <stdio.h>

__global__ void addValue(int* data, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int* ptr = reinterpret_cast<int*>(((char*)data) + 16 * tid);
  *ptr += value;
}

int main() {
  int numElements = 10;
  int value = 5;

  // Allocate memory on the host
  int* hostData = (int*)malloc(numElements * sizeof(int));

  // Initialize the host data
  printf("Before: ");
  for (int i = 0; i < numElements; i++) {
    hostData[i] = i;
    printf("%d ", hostData[i]);
  }
  printf("\n");

  // Allocate memory on the device
  int* deviceData;
  cudaMalloc((void**)&deviceData, numElements * sizeof(int));
  printf("Data mem addr: %p \n", deviceData);

  // Copy data from host to device
  cudaMemcpy(deviceData, hostData, numElements * sizeof(int), cudaMemcpyHostToDevice);

  // Launch the kernel
  int blockSize = 256;
  int gridSize = (numElements + blockSize - 1) / blockSize;
  addValue<<<gridSize, blockSize>>>(deviceData, value);
  addValue<<<gridSize, blockSize>>>(deviceData, value);
  addValue<<<gridSize, blockSize>>>(deviceData, value);
  addValue<<<gridSize, blockSize>>>(deviceData, value);
  addValue<<<gridSize, blockSize>>>(deviceData, value);

  // Copy data from device to host
  cudaMemcpy(hostData, deviceData, numElements * sizeof(int), cudaMemcpyDeviceToHost);

  // Print the updated data
  printf("After:  ");
  for (int i = 0; i < numElements; i++) {
    printf("%d ", hostData[i]);
  }
  printf("\n");

  // Free memory
  free(hostData);
  cudaFree(deviceData);

  return 0;
}
