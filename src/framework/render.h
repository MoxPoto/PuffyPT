#ifndef RENDER_H
#define RENDER_H

#include <Windows.h>
#include <thread>
#include <mutex>
#include <imgui.h>
#include <wrl.h>

#include <memory>

// #include <d3dx11.h>
#include <d3d11.h>
#include <d3d10.h>

using Microsoft::WRL::ComPtr;

extern void renderingFunc(ComPtr<ID3D11Device> device, ComPtr<IDXGISwapChain> swapChain, ComPtr<ID3D11DeviceContext> devContext, ComPtr<ID3D11RenderTargetView> backbuffer, std::mutex* renderMutex, ImFont* font);

#endif