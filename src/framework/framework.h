#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <Windows.h>
#include <thread>
#include <mutex>
#include <imgui.h>
#include <wrl/client.h>

#include <memory>

#include <d3d11.h>
#include <d3d10.h>

using Microsoft::WRL::ComPtr;

class Framework {
private:
	ComPtr<IDXGISwapChain> swapChain;
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> devContext;

	ComPtr<ID3D11RenderTargetView> backBuffer;

	std::thread renderer;
	std::mutex renderMutex;

	HWND window;
	ImFont* font;
public:
	void InitWindow();

	Framework();
	~Framework();
};

#endif