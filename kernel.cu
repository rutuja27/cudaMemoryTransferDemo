

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>

#include <time.h>
#include <ctime> //defines localtime 
#include <chrono>

#include <fstream>

#ifdef WIN32
#include<windows.h>
#endif

//using namespace System;
using namespace std;

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct TimeStamp
{
    unsigned long long int seconds;
    unsigned int microSeconds;
};

const int rows = 2048;
const int cols = 1200;



class GetTime
{

public:

    unsigned long long secs;
    unsigned long long usec;
    time_t curr_time;
    //timeval tv;

    GetTime(long long unsigned int secs, long long unsigned int usec);

    struct timezone
    {
        int  tz_minuteswest; /* minutes W of Greenwich */
        int  tz_dsttime;     /* type of dst correction */
    };

    // Definition of a gettimeofday function
    //int getdaytime(struct timeval *tv, struct timezone *tz);
    std::chrono::system_clock::duration duration_since_midnight();
    TimeStamp getPCtime();

};

GetTime::GetTime(long long unsigned int sec, long long unsigned int usec)
{
    secs = secs;
    usec = usec;
}

std::chrono::system_clock::duration GetTime::duration_since_midnight()
{

    auto now = std::chrono::system_clock::now();
    time_t tnow = std::chrono::system_clock::to_time_t(now);
    tm *date = std::localtime(&tnow);
    date->tm_hour = 0;
    date->tm_min = 0;
    date->tm_sec = 0;
    auto midnight = std::chrono::system_clock::from_time_t(std::mktime(date));
    return now - midnight;

}


TimeStamp GetTime::getPCtime()
{

    auto since_midnight = duration_since_midnight();

    auto hours = std::chrono::duration_cast<std::chrono::hours>(since_midnight);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(since_midnight - hours);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_midnight - hours - minutes);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(since_midnight - hours - minutes - seconds);
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(since_midnight - hours - minutes - seconds - milliseconds);
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(since_midnight - hours - minutes - seconds - milliseconds - microseconds);

    this->secs = (hours.count() * 3600 + minutes.count() * 60 + seconds.count());
    this->usec = (milliseconds.count() * 1000 + microseconds.count() + nanoseconds.count() / 1000);
    TimeStamp ts = { this->secs, uint64_t(this->usec) };

    return ts;
}

cudaError_t addWithCuda(unsigned int rows, unsigned int cols, short *c, 
                        short *a, short *b);


__global__ void addKernel(short *c, short *a, short *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{

    short *a = new short[rows*cols*100];
    short *b = new short[rows*cols*100];
    short *c = new short[rows*cols*100];
    unsigned int size = rows * cols * 100;

    for (int i = 0; i < size; i++) {

        a[i] = 0;
        b[i] = 0;
        c[i] = 0;
    }
    
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(rows, cols, c, a, b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        getchar();
        return 1;
    }
    else {
        fprintf(stderr, "addWithCuda Succeded");
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    getchar();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(unsigned int rows, unsigned int cols, short *c, short *a, short *b)
{
    short *dev_a = 0;
    short *dev_b = 0;
    short *dev_c = 0;
    cudaError_t cudaStatus;
    unsigned int size = rows * cols * 100;
    
    GetTime mem_time(0, 0);
    TimeStamp start, end;
    float elapsed;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(short));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(short));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;

    }
    start = mem_time.getPCtime();
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(short));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    end = mem_time.getPCtime();
    elapsed = ((end.seconds * 1e6 + end.microSeconds) - (start.seconds * 1e6 + start.microSeconds)) / 1e6;
    printf(" Time taken for mem alloc %lf", elapsed);

    start = mem_time.getPCtime();
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    end = mem_time.getPCtime();
    elapsed = (((end.seconds * 1e6) + end.microSeconds) - ((start.seconds * 1e6) + start.microSeconds)) / 1e6;
    printf(" Time taken for mem transfer host to device %lf\n", elapsed);

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    dim3 grid(640, 750);
    dim3 block(32, 16);
    addKernel<<<grid, block>>>(dev_c, dev_a, dev_b);
   
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    
    start = mem_time.getPCtime();
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(short), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    end = mem_time.getPCtime();
    elapsed = (((end.seconds * 1e6) + end.microSeconds) - ((start.seconds * 1e6) + start.microSeconds)) / 1e6;
    printf(" Time taken for mem transfer from dev to host %lf\n", elapsed);
    

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
