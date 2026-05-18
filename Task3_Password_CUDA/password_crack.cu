#include <stdio.h>
#include <cuda_runtime.h>

__global__ void decryptKernel(char *input, char *output, int shift, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length) {
        char c = input[idx];

        if (c >= 'A' && c <= 'Z') {
            output[idx] = ((c - 'A' - shift + 26) % 26) + 'A';
        }
        else if (c >= 'a' && c <= 'z') {
            output[idx] = ((c - 'a' - shift + 26) % 26) + 'a';
        }
        else {
            output[idx] = c;
        }
    }
}

int main() {
    FILE *file = fopen("encrypted.txt", "r");

    if (file == NULL) {
        printf("Cannot open encrypted.txt\n");
        return 1;
    }

    FILE *out = fopen("decrypted.txt", "w");

    if (out == NULL) {
        printf("Cannot create decrypted.txt\n");
        fclose(file);
        return 1;
    }

    char line[1024];

    while (fgets(line, sizeof(line), file)) {

        int length = 0;

        while (line[length] != '\0') {
            length++;
        }

        char *d_input;
        char *d_output;

        cudaMalloc((void**)&d_input, length);
        cudaMalloc((void**)&d_output, length);

        cudaMemcpy(d_input, line, length, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (length + threadsPerBlock - 1) / threadsPerBlock;

        decryptKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 3, length);

        cudaDeviceSynchronize();

        char output[1024];

        cudaMemcpy(output, d_output, length, cudaMemcpyDeviceToHost);

        fprintf(out, "%s", output);

        cudaFree(d_input);
        cudaFree(d_output);
    }

    fclose(file);
    fclose(out);

    printf("Password decryption completed. Output saved in decrypted.txt\n");

    return 0;
}
