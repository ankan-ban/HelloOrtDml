// Minimal C++ example for using Onnxruntime APIs with DML ep
// 
// Goals:
//   - Avoid serial CPU <-> GPU transfers at each inference.
//     * If really needed, demonstrate how to use asynchronous d3d12 copy queues to handle the transfers
//   - Pipeline multiple inference requests to keep GPU occupied all the time (i.e, don't wait immediately for inference result).
// 

const bool useCpuBindings = false;          // bind resources on CPU memory
const bool perIterationTransfers = true;    // Set to true to perform CPU<->GPU transfers in each iteration (pipelined with inference work)
constexpr int iterationInFlight = 3;        // no of "iterations in flight" for the pipeline. 1 means no parallelism.
constexpr bool useFp16Model = true;

// for benchmarking
const int warmupIterations = 1000;
const int iterations = 1000;

#include <stdio.h>

#include "Common.h"
#include "dml_provider_factory.h"
#include "onnxruntime_cxx_api.h"
#include <chrono>

struct GpuResourceData
{
    ID3D12Resource* pInput;
    ID3D12Resource* pOutput;

    ID3D12Resource* pUploadRes;
    ID3D12Resource* pDownloadRes;

    // command lists to upload/download the inputs/outputs
    ID3D12GraphicsCommandList* pUploadCommandList;
    ID3D12GraphicsCommandList* pDownloadCommandList;

    void* dml_resource_input;
    void* dml_resource_output;
};

float cpuInputFloat[3 * 720 * 720];
float cpuOutputFloat[3 * 720 * 720];

uint16_t cpuInputHalf[3 * 720 * 720];
uint16_t cpuOutputHalf[3 * 720 * 720];

int main()
{
    HRESULT hr = S_OK;

    ID3D12Device* pDevice;
    ID3D12CommandQueue* pCommandQueue;
    ID3D12CommandQueue* pUploadQueue;
    ID3D12CommandQueue* pDownloadQueue;
    ID3D12CommandAllocator* pAllocator;

    GpuResourceData resources[iterationInFlight];

    // load the input image from file into CPU memory
    if (useFp16Model)
        loadInputImage(cpuInputHalf, "input.png", true);
    else
        loadInputImage(cpuInputFloat, "input.png", false);

    // Create Device
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));

    // command allocator to record upload / download commands
    hr = pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&pAllocator));

    // create the additional command queues to manage async uploads and downloads
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pUploadQueue));
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pDownloadQueue));

    int numCopiesToCreate = 1;
    if (perIterationTransfers)
    {
        numCopiesToCreate = iterationInFlight;
    }

    // Create d3d12 resources (to be used for input and output of the network)
    for (int i = 0; i < numCopiesToCreate; i++)
    {
        // default resources
        CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &resources[i].pInput, D3D12_RESOURCE_STATE_COPY_DEST);
        CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &resources[i].pOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);

        // upload and download resources
        CreateUploadBuffer(pDevice, 3 * 720 * 720 * sizeof(float), &resources[i].pUploadRes);
        CreateReadBackBuffer(pDevice, 3 * 720 * 720 * sizeof(float), &resources[i].pDownloadRes);

        // command lists to handle uploads/downloads

        // record the commands in the upload command list
        hr = pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COPY, pAllocator, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&resources[i].pUploadCommandList);
        resources[i].pUploadCommandList->CopyResource(resources[i].pInput, resources[i].pUploadRes);
        resources[i].pUploadCommandList->Close();

        // record the commands in the download command list
        hr = pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COPY, pAllocator, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&resources[i].pDownloadCommandList);
        resources[i].pDownloadCommandList->CopyResource(resources[i].pDownloadRes, resources[i].pOutput);
        resources[i].pDownloadCommandList->Close();
    }

    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFenceUpload = nullptr;
    ID3D12Fence* pFenceInference = nullptr;
    ID3D12Fence* pFenceDownload = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFenceUpload));
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFenceInference));
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFenceDownload));

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

    Ort::Session session = Ort::Session(ortEnvironment, useFp16Model ? L"fns-candy-fp16.onnx" : L"fns-candy.onnx", sessionOptions);

    Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);
    Ort::MemoryInfo memoryInformation("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);
    Ort::Allocator deviceAllocator(session, memoryInformation);

    // we know there is only single input and single output for this model
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr OuptutTensorName = session.GetOutputNameAllocated(0, allocator);

    int64_t inputDim[] = { 1, 3, 720, 720 };
    int64_t outputDim[] = { 1, 3, 720, 720 };

    // Create ORT tensors from D3D12 resources that we created
    for (int i = 0; i < numCopiesToCreate; i++)
    {
        ortDmlApi->CreateGPUAllocationFromD3DResource(resources[i].pInput, &resources[i].dml_resource_input);
        ortDmlApi->CreateGPUAllocationFromD3DResource(resources[i].pOutput, &resources[i].dml_resource_output);
    }


    // CPU based binding path
    OrtValue* cpu_ort_tensor_input = NULL;
    OrtValue* cpu_ort_tensor_output = NULL;

    if (useCpuBindings)
    {
        /*
        OrtMemoryInfo* cpu_memory_info;
        ortApi.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info);
        if (useFp16Model)
        {
            ortApi.CreateTensorWithDataAsOrtValue(cpu_memory_info, cpuInputFloat, sizeof(cpuInputFloat), inputDim,
                4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cpu_ort_tensor_input);
            ortApi.CreateTensorWithDataAsOrtValue(cpu_memory_info, cpuOutputFloat, sizeof(cpuOutputFloat), inputDim,
                4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &cpu_ort_tensor_output);
        }
        else
        {
            ortApi.CreateTensorWithDataAsOrtValue(cpu_memory_info, cpuInputHalf, sizeof(cpuInputHalf), inputDim,
                4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &cpu_ort_tensor_input);
            ortApi.CreateTensorWithDataAsOrtValue(cpu_memory_info, cpuOutputHalf, sizeof(cpuOutputHalf), inputDim,
                4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &cpu_ort_tensor_output);
        }
        */
        OrtMemoryInfo* cpu_memory_info;
        ortApi.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info);

        Ort::Value inputTensor(Ort::Value::CreateTensor(cpu_memory_info, 
            useFp16Model ? (void*) cpuInputHalf : (void *)cpuInputFloat, 
            useFp16Model ? sizeof(cpuInputHalf) : sizeof(cpuInputFloat),
            inputDim, 4, 
            useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        Ort::Value outputTensor(Ort::Value::CreateTensor(cpu_memory_info, 
            useFp16Model ? (void*)cpuOutputHalf : (void*)cpuOutputFloat,
            useFp16Model ? sizeof(cpuOutputHalf) : sizeof(cpuOutputFloat),
            outputDim, 4, 
            useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        ioBinding.BindInput(InputTensorName.get(), inputTensor);
        ioBinding.BindOutput(OuptutTensorName.get(), outputTensor);
        ioBinding.SynchronizeInputs();
    }
    else if (!perIterationTransfers)
    {
        // upload the input and wait for the upload to finish
        void* pData;
        resources[0].pUploadRes->Map(0, nullptr, (void**)&pData);
        if (useFp16Model)
            memcpy(pData, cpuInputHalf, sizeof(cpuInputHalf));
        else
            memcpy(pData, cpuInputFloat, sizeof(cpuInputFloat));
        resources[0].pUploadRes->Unmap(0, nullptr);

        pUploadQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&resources[0].pUploadCommandList);
        FlushAndWait(pDevice, pUploadQueue);

        // bind the resources
        Ort::Value inputTensor(Ort::Value::CreateTensor(memoryInformation, resources[0].dml_resource_input, resources[0].pInput->GetDesc().Width,
            inputDim, 4, useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        Ort::Value outputTensor(Ort::Value::CreateTensor(memoryInformation, resources[0].dml_resource_output, resources[0].pOutput->GetDesc().Width,
            outputDim, 4, useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        ioBinding.BindInput(InputTensorName.get(), inputTensor);
        ioBinding.BindOutput(OuptutTensorName.get(), outputTensor);
        ioBinding.SynchronizeInputs();
    }
    // Benchmark the model (schedule 100 iterations on the command queue for testing)
    Ort::RunOptions runOptions;

    // benchmarking run with few warm-up iterations
    auto start = std::chrono::high_resolution_clock::now();
    int totalIterations = warmupIterations + iterations;
    for (int i = 0; i < totalIterations; i++)
    {
        int fenceVal = i + 1;

        if (i == warmupIterations)
            start = std::chrono::high_resolution_clock::now();

        if (useCpuBindings)
        {
            session.Run(runOptions, ioBinding);
        }
        else
        {
            int resourceIndex = fenceVal % numCopiesToCreate;

            if (perIterationTransfers)
            {
                // copy input from CPU->GPU
                void* pData;
                resources[resourceIndex].pUploadRes->Map(0, nullptr, (void**)&pData);
                if (useFp16Model)
                    memcpy(pData, cpuInputHalf, sizeof(cpuInputHalf));
                else
                    memcpy(pData, cpuInputFloat, sizeof(cpuInputFloat));
                resources[resourceIndex].pUploadRes->Unmap(0, nullptr);

                pUploadQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&resources[resourceIndex].pUploadCommandList);
                //FlushAndWait(pDevice, pUploadQueue);   // For debug
                pUploadQueue->Signal(pFenceUpload, fenceVal);

                // Bind the inputs and outputs
                Ort::Value inputTensor(Ort::Value::CreateTensor(memoryInformation, resources[resourceIndex].dml_resource_input, resources[resourceIndex].pInput->GetDesc().Width,
                    inputDim, 4, useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
                Ort::Value outputTensor(Ort::Value::CreateTensor(memoryInformation, resources[resourceIndex].dml_resource_output, resources[resourceIndex].pOutput->GetDesc().Width,
                    outputDim, 4, useFp16Model ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

                ioBinding.BindInput(InputTensorName.get(), inputTensor);
                ioBinding.BindOutput(OuptutTensorName.get(), outputTensor);

                // make the inference wait for the upload
                pCommandQueue->Wait(pFenceUpload, fenceVal);
            }

            session.Run(runOptions, ioBinding);
            pCommandQueue->Signal(pFenceInference, fenceVal);

            if (perIterationTransfers)
            {
                // Make the download wait for the inference
                pDownloadQueue->Wait(pFenceInference, fenceVal);

                // copy output from GPU->CPU
                pDownloadQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&resources[resourceIndex].pDownloadCommandList);
                // FlushAndWait(pDevice, pDownloadQueue);   // debug
                pDownloadQueue->Signal(pFenceDownload, fenceVal);

            }


            // wait for (i-2)nd iteration (so that we have 2 iterations in flight)
            int oldIter = fenceVal - (iterationInFlight - 1);

            if (oldIter > 0)
            {
                if (perIterationTransfers)
                {
                    pFenceDownload->SetEventOnCompletion(oldIter, hEvent);
                    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

                    resourceIndex = oldIter % numCopiesToCreate;

                    // read back data
                    void* pData;
                    resources[resourceIndex].pDownloadRes->Map(0, nullptr, (void**)&pData);
                    if (useFp16Model)
                        memcpy(cpuOutputHalf, pData, sizeof(cpuOutputHalf));
                    else
                        memcpy(cpuOutputFloat, pData, sizeof(cpuOutputFloat));
                    resources[resourceIndex].pDownloadRes->Unmap(0, nullptr);
                }
                else
                {
                    pFenceInference->SetEventOnCompletion(oldIter, hEvent);
                    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
                }
            }
        }
    }

    // Wait for the last iteration
    if (!useCpuBindings)
    {
        pFenceInference->SetEventOnCompletion(totalIterations, hEvent);
        DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    if (!useCpuBindings && !perIterationTransfers)
    {
        // download the output to cpu memory
        pDownloadQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&resources[0].pDownloadCommandList);
        FlushAndWait(pDevice, pDownloadQueue);

        void* pData;
        resources[0].pDownloadRes->Map(0, nullptr, (void**)&pData);
        if (useFp16Model)
            memcpy(cpuOutputHalf, pData, sizeof(cpuOutputHalf));
        else
            memcpy(cpuOutputFloat, pData, sizeof(cpuOutputFloat));
        resources[0].pDownloadRes->Unmap(0, nullptr);
    }

    // save the output to disk
    if (useFp16Model)
        saveOutputImage(cpuOutputHalf, "output.png", true);
    else
        saveOutputImage(cpuOutputFloat, "output.png", false);

    for (int i = 0; i < numCopiesToCreate; i++)
    {
        ortDmlApi->FreeGPUAllocation(resources[i].dml_resource_input);
        ortDmlApi->FreeGPUAllocation(resources[i].dml_resource_output);

        resources[i].pInput->Release();
        resources[i].pOutput->Release();

        resources[i].pDownloadCommandList->Release();
        resources[i].pUploadCommandList->Release();
        resources[i].pUploadRes->Release();
        resources[i].pDownloadRes->Release();
    }

    if (cpu_ort_tensor_input)
        ortApi.ReleaseValue(cpu_ort_tensor_input);
    if (cpu_ort_tensor_output)
        ortApi.ReleaseValue(cpu_ort_tensor_output);

    pDmlDevice->Release();
    pFenceUpload->Release();
    pFenceInference->Release();
    pFenceDownload->Release();
    pCommandQueue->Release();
    pUploadQueue->Release();
    pDownloadQueue->Release();
    pAllocator->Release();
    pDevice->Release();
    session.release();
    CloseHandle(hEvent);
    printf("\nInference loop done. %d iterations in %g ms - avg: %g ms per iteration\n", iterations, duration, duration/iterations);
    return 0;
}