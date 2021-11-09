// Takes care of setting up the shader resources

#include <framework/framework.h>

#include <d3d11.h>
#include <d3d10.h>
// #include <d3dx11.h>

#include <d3dcommon.h>

#include <Windows.h>
#include <stdio.h>

#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx11.h>
#include <globals.h>

#include <filesystem>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

void Framework::PrepareFramebuffer() {
	D3D11_BUFFER_DESC fbDesc;
	ZeroMemory(&fbDesc, sizeof(fbDesc));

	fbDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
	fbDesc.ByteWidth = width * height * (sizeof(float) * 3);
	fbDesc.Usage = D3D11_USAGE_DEFAULT;
	fbDesc.CPUAccessFlags = 0; // The CPU doesn't really need access to the framebuffer
	fbDesc.StructureByteStride = sizeof(float) * 3;
	fbDesc.MiscFlags = 0;

	device->CreateBuffer(&fbDesc, NULL, fbBuffer.GetAddressOf());

	if (!fbBuffer) {
		printf("Failed to create the framebuffer buffer!\n");
	}

	D3D11_UNORDERED_ACCESS_VIEW_DESC fbUAVDesc;
	ZeroMemory(&fbUAVDesc, sizeof(fbUAVDesc));

	fbUAVDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
	fbUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	fbUAVDesc.Buffer.FirstElement = 0;
	fbUAVDesc.Buffer.Flags = 0;
	fbUAVDesc.Buffer.NumElements = width * height;

	device->CreateUnorderedAccessView(fbBuffer.Get(), &fbUAVDesc, fbUAV.GetAddressOf());

	if (!fbUAV) {
		printf("Failed to create the framebuffer UAV!");
	}
}