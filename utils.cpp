#include "Common.h"
#include "lodepng/lodepng.h"

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

static void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue)
{
    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFence = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

    pQueue->Signal(pFence, 1);
    pFence->SetEventOnCompletion(1, hEvent);
    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

    pFence->Release();
    CloseHandle(hEvent);
}


HRESULT uploadInputImageToD3DResource(ID3D12Device *pDevice, ID3D12CommandQueue *pQueue, ID3D12Resource* pResource, char* imageFileName)
{
    unsigned char* image;
    unsigned int width, height;
    unsigned int error = lodepng_decode32_file(&image, &width, &height, imageFileName);
    if (error) {
        printf("\nFailed to load the input image. Exiting\n");
        exit(0);
    }

    if (width != 720 || height != 720) {
        printf("\nImage not of right size (720x720). Exiting\n");
        exit(0);
    }

    // Create upload resource
    ID3D12Resource* pUploadRes = nullptr;
    ID3D12Resource* g_pBufferUpload;
    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(3 * width * height * sizeof(float)),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&pUploadRes));

    float* pData;   // CHW data in BGR order
    pUploadRes->Map(0, nullptr, (void**)&pData);
    for (int y=0;y<height;y++)
        for (int x = 0; x < width; x++)
        {
            unsigned char r = image[(y * width + x) * 4 + 0];
            unsigned char g = image[(y * width + x) * 4 + 1];
            unsigned char b = image[(y * width + x) * 4 + 2];

            pData[0 * width * height + y * width + x] = (float)b;
            pData[1 * width * height + y * width + x] = (float)g;
            pData[2 * width * height + y * width + x] = (float)r;
        }
    pUploadRes->Unmap(0, nullptr);
    free(image);

    // Create Command allocator and command list
    ID3D12CommandAllocator* pAllocator;
    hr = pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pAllocator));
    ID3D12GraphicsCommandList* pCL;
    hr = pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT, pAllocator, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&pCL);

    pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST));
    pCL->CopyResource(pResource, pUploadRes);
    pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    pCL->Close();
    pQueue->ExecuteCommandLists(1, (ID3D12CommandList **)&pCL);
    FlushAndWait(pDevice, pQueue);
    pAllocator->Reset();
    pCL->Reset(pAllocator, nullptr);

    pAllocator->Release();
    pCL->Release();
    pUploadRes->Release();

    return S_OK;
}

unsigned char clampAndConvert(float val)
{
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (unsigned char)val;
}

HRESULT saveOutputImageFromD3DResource(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue, ID3D12Resource* pResource, char* imageFileName)
{
    unsigned int width = 720, height = 720; // hardcoded in the model

    // Create the GPU readback buffer.
    ID3D12Resource* pReadbackRes;
    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(3 * width * height * sizeof(float)),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&pReadbackRes));


    // Create Command allocator and command list
    ID3D12CommandAllocator* pAllocator;
    hr = pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&pAllocator));
    ID3D12GraphicsCommandList* pCL;
    hr = pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT, pAllocator, NULL, __uuidof(ID3D12GraphicsCommandList), (void**)&pCL);

    pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE));
    pCL->CopyResource(pReadbackRes, pResource);
    pCL->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(pResource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    pCL->Close();
    pQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&pCL);

    FlushAndWait(pDevice, pQueue);
    pAllocator->Reset();
    pCL->Reset(pAllocator, nullptr);
    pAllocator->Release();
    pCL->Release();

    std::vector<unsigned char> image(width*height*4);
    float* pData;   // CHW data
    pReadbackRes->Map(0, nullptr, (void**)&pData);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float b = pData[0 * width * height + y * width + x];
            float g = pData[1 * width * height + y * width + x];
            float r = pData[2 * width * height + y * width + x];

            image[(y * width + x) * 4 + 0] = clampAndConvert(r);
            image[(y * width + x) * 4 + 1] = clampAndConvert(g);
            image[(y * width + x) * 4 + 2] = clampAndConvert(b);
            image[(y * width + x) * 4 + 3] = 255;
        }
    pReadbackRes->Unmap(0, nullptr);
    pReadbackRes->Release();

    lodepng_encode32_file(imageFileName, &image[0], width, height);
}