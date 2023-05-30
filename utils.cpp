#include "Common.h"
#include "lodepng/lodepng.h"

void CreateD3D12Buffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource, D3D12_RESOURCE_STATES initState)
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
        initState,
        nullptr,
        IID_PPV_ARGS(ppResource));

    if (FAILED(hr))
    {
        printf("\nFailed creating a resource\n");
        exit(0);
    }
}

void CreateUploadBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource)
{
    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(size),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(ppResource));
}

void CreateReadBackBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource)
{
    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(size),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(ppResource));
}


void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue)
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


void loadInputImage(float *pData, char* imageFileName)
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

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            unsigned char r = image[(y * width + x) * 4 + 0];
            unsigned char g = image[(y * width + x) * 4 + 1];
            unsigned char b = image[(y * width + x) * 4 + 2];

            pData[0 * width * height + y * width + x] = (float)b;
            pData[1 * width * height + y * width + x] = (float)g;
            pData[2 * width * height + y * width + x] = (float)r;
        }

    free(image);
}

unsigned char clampAndConvert(float val)
{
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (unsigned char)val;
}

void saveOutputImage(float *pData, char* imageFileName)
{
    unsigned int width = 720, height = 720; // hardcoded in the model

    std::vector<unsigned char> image(width * height * 4);
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

    lodepng_encode32_file(imageFileName, &image[0], width, height);
}