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

#include <framework/shaders.h>

using Microsoft::WRL::ComPtr;

class Framework {
private:
	ComPtr<IDXGISwapChain> swapChain;
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> devContext;

	ComPtr<ID3D11RenderTargetView> backBuffer;

	std::thread renderer;
	std::mutex renderMutex;

	std::unique_ptr<Shaders> shaderService;

	HWND window;
	ImFont* font;
public:
	int width;
	int height;

	ComPtr<ID3D11ComputeShader> pathtracer;

	ComPtr<ID3D11Buffer> fbBuffer;
	ComPtr<ID3D11UnorderedAccessView> fbUAV;

	void InitWindow();
	void CompilePuffyPT();
	void Error(const char* title, const char* errorMsg, UINT type);
	void PrepareFramebuffer();

	Framework();
	~Framework();
};

#endif