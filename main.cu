#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

__device__ float customSqrt(float x) {
    float xhalf = 0.5f * x;
    int i = *(int*)&x;            // get bits for floating VALUE
    i = 0x5f3759df - (i >> 1);     // gives initial guess y0
    x = *(float*)&i;               // convert bits BACK to float
    x = x * (1.5f - xhalf * x * x); // Newton step, repeating increases accuracy
    return 1 / x;
}

__global__ void sobelFilterCUDA(const uchar* input, uchar* output, int rows, int cols, int scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int offset = y * cols + x;

        int gx = 0, gy = 0;

        gx = -1 * input[(y - scale) * cols + (x - scale)] + 2 * input[(y - scale) * cols + x] + input[(y - scale) * cols + (x + scale)] +
             -1 * input[(y + scale) * cols + (x - scale)] + 2 * input[(y + scale) * cols + x] + input[(y + scale) * cols + (x + scale)];

        output[offset] = customSqrt(gx * gx + gy * gy);
    }
}

void sobelFilter(const Mat& inputImage, Mat& outputImage, int scale) {
    int rows = inputImage.rows;
    int cols = inputImage.cols;

    Size imageSize = inputImage.size();
    int imageSizeBytes = inputImage.step * imageSize.height;

    uchar* d_inputImage;
    uchar* d_outputImage;
    cudaMalloc((void**)&d_inputImage, imageSizeBytes);
    cudaMalloc((void**)&d_outputImage, imageSizeBytes);

    cudaMemcpy(d_inputImage, inputImage.data, imageSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    sobelFilterCUDA<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, rows, cols, scale);

    outputImage.create(rows, cols, CV_8U);
    cudaMemcpy(outputImage.data, d_outputImage, imageSizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main() {
    Mat inputImage = imread("input_image.png", IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        cerr << "Failed to load input image." << endl;
        return -1;
    }

    Mat outputImage1, outputImage2, outputImage3;

    sobelFilter(inputImage, outputImage1, 1);
    sobelFilter(inputImage, outputImage2, 2);
    sobelFilter(inputImage, outputImage3, 3);

    imwrite("output1.png", outputImage1);
    cout << "Image1 processed and saved successfully." << endl;

    imwrite("output2.png", outputImage2);
    cout << "Image2 processed and saved successfully." << endl;

    imwrite("output3.png", outputImage3);
    cout << "Image3 processed and saved successfully." << endl;

    return 0;
}