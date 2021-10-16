#include <framework/framework.h>
#include <framework/window.h>
#include <framework/render.h>

#include <d3d9.h>
#include <d3dx9.h>

#include <Windows.h>
#include <stdio.h>

#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx11.h>
#include <globals.h>

#include <wrl/client.h>


using Microsoft::WRL::ComPtr;
// renderingFunc is located in framework/render.h

void Framework::InitWindow() {
	// Create a window from the gmod process
	window = CreatePuffyWindow();

	if (window == NULL) {
		// TODO: Figure out some error mechanic
		printf("Couldn't create the Puffy PT window!");
	}

	DXGI_SWAP_CHAIN_DESC description;
	ZeroMemory(&description, sizeof(DXGI_SWAP_CHAIN_DESC));

	description.BufferCount = 1; // Just one
	description.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	description.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	description.OutputWindow = window;
	description.SampleDesc.Count = 4; // 4 AA samples
	description.Windowed = TRUE;
	description.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
	
	// Create our device and swapchain
	HRESULT code = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, NULL, D3D11_SDK_VERSION, &description, swapChain.GetAddressOf(), device.GetAddressOf(), NULL, devContext.GetAddressOf());
	if (code != S_OK) {
		// Something went really wrong
		printf("Couldn't create D3D11 Device and Swapchain!!!\n");
	}
}

Framework::Framework() {
	// Initialize some critical services
	AllocConsole();
	FILE* pFile = nullptr;

	freopen_s(&pFile, "CONOUT$", "w", stdout); // cursed way to redirect stdout to our own console
	
	InitWindow();
	// Setup the backbuffer
	ComPtr<ID3D11Texture2D> backbufferTex;
	swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(backbufferTex.GetAddressOf()));

	device->CreateRenderTargetView(backbufferTex.Get(), NULL, backBuffer.GetAddressOf());
	backbufferTex->Release();

	// Set the rendertarget to our backbuffer
	devContext->OMSetRenderTargets(1, backBuffer.GetAddressOf(), NULL);

	// And finally, setup the viewport
	D3D11_VIEWPORT viewport;
	ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));

	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;
	viewport.Width = 1728;
	viewport.Height = 972;

	devContext->RSSetViewports(1, &viewport); // Done, ready for rendering

	// Sick, now let's create ImGui
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	
	ImGuiStyle& style = ImGui::GetStyle();

	style.FrameRounding = 5.0f;
	style.ChildRounding = 5.0f;
	style.WindowRounding = 5.0f;
	style.WindowTitleAlign = ImVec2(0.5, 0.5);
	style.Colors[ImGuiCol_TitleBg] = ImVec4(0.592, 0.090, 0.090, 1);
	style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.392, 0.020, 0.020, 1);

	font = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\micross.ttf", 15);

	ImGui::StyleColorsDark();
	ImGui_ImplWin32_Init(window);
	ImGui_ImplDX11_Init(device.Get(), devContext.Get());

	// Set our alive
	alive = true;
	// Validate we have compute shader support
	if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0) {
		D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts = { 0 };
		(void)device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts));
		if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x)
		{
			printf("DirectCompute is not supported! Not running the framework..");
			alive = false;
		}
	}

	renderer = std::thread(renderingFunc, device, swapChain, devContext, backBuffer, &renderMutex, font);
	renderer.detach();
}

Framework::~Framework() {
	renderMutex.lock(); // Lock real quick

	printf("Locking..\n");
	// Close the main window
	// And free the console

	// Get ImGui closed too
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	printf("Destroyed ImGui..\n");

	// Cleanup D3D resources
	// And switch out of fullscreen
	swapChain->SetFullscreenState(FALSE, NULL);

	swapChain->Release();
	backBuffer->Release();
	device->Release();
	devContext->Release();

	printf("Cleaned D3D..\n");

	if (window != NULL) {
		DestroyWindow(window);
	}

	// Also, unregister the window class we created
	UnregisterClass(PUFFYPT_CLASS, GetModuleHandle(NULL));
	// The goal when closing, is to essentially act as if nothing ever happened, if we leave some type of trace (eg. memory leak or leftover resources),
	// then those could actually result in a crash if we open the module more than once

	printf("Removed window..\n");
	alive = false;
	FreeConsole();
	
	renderMutex.unlock();
}