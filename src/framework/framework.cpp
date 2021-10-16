#include <framework/framework.h>
#include <framework/window.h>
#include <framework/render.h>
#include <pathtracer/pathtracer.cuh>

#include <d3d9.h>
#include <d3dx9.h>

#include <Windows.h>
#include <stdio.h>

#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx9.h>
#include <globals.h>

// renderingFunc is located in framework/render.h

void Framework::InitWindow() {
	// Create a window from the gmod process
	window = CreatePuffyWindow();

	if (window == NULL) {
		// TODO: Figure out some error mechanic
		printf("Couldn't create the Puffy PT window!");
	}

	d3d = Direct3DCreate9(D3D_SDK_VERSION);
	// Create a rendering context
	ZeroMemory(&presentParams, sizeof(presentParams)); // Clear it out for our usage

	presentParams.Windowed = true;
	presentParams.SwapEffect = D3DSWAPEFFECT_DISCARD;
	presentParams.hDeviceWindow = window;

	d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING, &presentParams, &device);
}

void Framework::SetPathtracer(std::shared_ptr<Pathtracer> pathtracerPtr) {
	pathtracer = pathtracerPtr;
}

Framework::Framework() {
	// Initialize some critical services
	AllocConsole();
	FILE* pFile = nullptr;

	freopen_s(&pFile, "CONOUT$", "w", stdout); // cursed way to redirect stdout to our own console
	
	InitWindow();
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
	ImGui_ImplDX9_Init(device);

	renderSprite = NULL;
	HRESULT code = D3DXCreateSprite(device, &renderSprite);
	
	if (!renderSprite) {
		printf("Couldn't generate renderSprite!\nCode: %u\n", static_cast<unsigned int>(code));
		return;
	}

	pathtracer = std::make_shared<Pathtracer>(1728, 972, device);

	// Set our alive
	alive = true;

	std::shared_ptr<ID3DXSprite> ptr(renderSprite);

	renderer = std::thread(renderingFunc, device, &renderMutex, font, pathtracer, ptr);
	renderer.detach();
}

Framework::~Framework() {
	renderMutex.lock(); // Lock real quick

	printf("Locking..\n");
	// Close the main window
	// And free the console

	// Get ImGui closed too
	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	printf("Destroyed ImGui..\n");

	// Cleanup D3D resources

	device->Release();
	d3d->Release();

	printf("Cleaned D3D..\n");

	if (window != NULL) {
		DestroyWindow(window);
	}

	// Remove pathtracer
	pathtracer.reset();

	// Also, unregister the window class we created
	UnregisterClass(PUFFYPT_CLASS, GetModuleHandle(NULL));
	// The goal when closing, is to essentially act as if nothing ever happened, if we leave some type of trace (eg. memory leak or leftover resources),
	// then those could actually result in a crash if we open the module more than once

	printf("Removed window..\n");
	alive = false;
	FreeConsole();
	
	renderMutex.unlock();
}