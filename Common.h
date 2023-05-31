#pragma once
#include "d3dx12.h"

void CreateD3D12Buffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource, D3D12_RESOURCE_STATES initState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
void CreateUploadBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource);
void CreateReadBackBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource);

void loadInputImage(void* pData, char* imageFileName, bool fp16);
void saveOutputImage(void* pData, char* imageFileName, bool fp16);

void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue);
