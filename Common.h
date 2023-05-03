#pragma once
#include "d3dx12.h"

void CreateD3D12Buffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource);

HRESULT uploadInputImageToD3DResource(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue, ID3D12Resource* pResource, char* imageFileName);
HRESULT saveOutputImageFromD3DResource(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue, ID3D12Resource* pResource, char* imageFileName);