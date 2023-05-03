// Minimal C++ example for using Onnxruntime APIs with DML ep
// 
// Goals:
//   - Avoid CPU <-> GPU transfers at each inference (TODO: also demonstrate how to use d3d12 copy queue to pipeline the copies)
//   - pipeline multiple inference requests to keep GPU occupied all the time.

const int warmupIterations = 100;
const int iterations = 100;

#include <stdio.h>

#include "d3dx12.h"

#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"
#include <chrono>

void CreateD3D12Buffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource)
{
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.SampleDesc.Quality = 0;
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(ppResource));

    if (FAILED(hr))
    {
        printf("\nFailed creating a resource\n");
        exit(0);
    }
}


int main()
{
    HRESULT hr = S_OK;

    ID3D12Device* pDevice;
    ID3D12CommandQueue* pCommandQueue;
    ID3D12Resource* pInput;
    ID3D12Resource* pOutput;

    // Create Device
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));

    // Create d3d12 resources (to be used for input and output of the network)
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pInput);
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pOutput);

    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFence = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

    // DML device to use for ORT APIs
    IDMLDevice* pDmlDevice = nullptr;
    hr = DMLCreateDevice(pDevice, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&pDmlDevice));

    // Use ORT APIs to load the model
    OrtApi const& ortApi = Ort::GetApi();
    const OrtDmlApi* ortDmlApi;
    auto ortStatus = ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ortDmlApi));
    Ort::Env ortEnvironment(ORT_LOGGING_LEVEL_WARNING, "HelloOrtDml");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.DisableMemPattern();
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Important - always specify/override all named dimensions
    // (By default they are set to 1, but DML optimizations get turned off if a model
    //  has dynamic dimensions that are not explicitly specified).
    ortApi.AddFreeDimensionOverrideByName(sessionOptions, "None", 1);

    // Make ORT use DML EP
    ortDmlApi->SessionOptionsAppendExecutionProvider_DML1(sessionOptions, pDmlDevice, pCommandQueue);

    Ort::Session session = Ort::Session(ortEnvironment, L"fns-candy.onnx", sessionOptions);

    Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);
    Ort::MemoryInfo memoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::Allocator deviceAllocator(session, memoryInformation);

    printf("Found %d inputs and %d outputs in the model", (int)session.GetInputCount(), (int)session.GetOutputCount());

    // we know there is only single input and single output for this model
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr OuptutTensorName = session.GetOutputNameAllocated(0, allocator);

    printf("\nInput Tensor Name: %s, output tensor name: %s\n", InputTensorName.get(), OuptutTensorName.get());

    Ort::TypeInfo InputTypeInfo = session.GetInputTypeInfo(0);
    Ort::TypeInfo OutputTypeInfo = session.GetOutputTypeInfo(0);

    int64_t inputDim[] = { 1, 3, 720, 720 };
    int64_t outputDim[] = { 1, 3, 720, 720 };

    // Create ORT tensors from D3D12 resources that we created, and bind them.
    void* dml_resource_input;
    ortDmlApi->CreateGPUAllocationFromD3DResource(pInput, &dml_resource_input);
    Ort::Value inputTensor(
        Ort::Value::CreateTensor(
            memoryInformation,
            dml_resource_input,
            pInput->GetDesc().Width,
            inputDim,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        )
    );

    void* dml_resource_output;
    ortDmlApi->CreateGPUAllocationFromD3DResource(pOutput, &dml_resource_output);
    Ort::Value outputTensor(
        Ort::Value::CreateTensor(
            memoryInformation,
            dml_resource_output,
            pOutput->GetDesc().Width,
            outputDim,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        )
    );

    ioBinding.BindInput(InputTensorName.get(), inputTensor);
    ioBinding.BindOutput(OuptutTensorName.get(), outputTensor);
    ioBinding.SynchronizeInputs();

    // Benchmark the model (schedule 100 iterations on the command queue for testing)
    Ort::RunOptions runOptions;

    // Warmup
    for (int i = 1; i <= warmupIterations; i++)
    {
        session.Run(runOptions, ioBinding);
        pCommandQueue->Signal(pFence, i);
        pFence->SetEventOnCompletion(i, hEvent);    // immediately wait for the GPU results
        DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
    }

    // Actual run for benchmarking

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        session.Run(runOptions, ioBinding);
        pCommandQueue->Signal(pFence, i + 1);

        // wait for (i-2)nd iteration (so that we have 2 iterations in flight)
        if (i > 1)
        {
            pFence->SetEventOnCompletion(i - 1, hEvent);
            DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
        }
    }

    // Wait for the last iteration
    pFence->SetEventOnCompletion(iterations, hEvent);
    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    ortDmlApi->FreeGPUAllocation(dml_resource_input);
    ortDmlApi->FreeGPUAllocation(dml_resource_output);
    pDmlDevice->Release();
    pFence->Release();
    pInput->Release();
    pOutput->Release();
    pCommandQueue->Release();
    pDevice->Release();
    session.release();
    printf("\nInference loop done. %d iterations in %g ms - avg: %g ms per iteration\n", iterations, duration, duration/iterations);
    return 0;
}