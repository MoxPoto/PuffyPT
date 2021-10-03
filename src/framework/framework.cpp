#include <framework/framework.h>
#include <framework/window.h>

#include <d3d9.h>
#include <Windows.h>
#include <stdio.h>

#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx9.h>

// There should be some way to make this.. better
static bool alive = true;
static bool test = true;

// TODO: Export this out of framework.cpp
static void renderingFunc(LPDIRECT3DDEVICE9 device, std::mutex* renderMutex) {
	while (alive) {
		// Dont ask
		renderMutex->lock();

		// Double check since we may've just terminated
		if (alive == false) {
			// Terminate
			break;
		}

		device->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 40, 100), 1.0f, 0);
		device->BeginScene();

		// Start a new ImGui frame
		ImGui_ImplDX9_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		ImGui::ShowDemoWindow(&test);

		// End it
		ImGui::EndFrame();
		device->SetRenderState(D3DRS_ZENABLE, FALSE);
		device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
		device->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);

		ImGui::Render();
		ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());

		device->EndScene();
		device->Present(NULL, NULL, NULL, NULL);

		renderMutex->unlock();
	}
}

void Framework::InitWindow() {
	// Create a window from the gmod process
	window = CreatePuffyWindow();

	if (window == NULL) {
		// TODO: Figure out some error mechanic
		printf("Couldn't create the Puffy PT window!");
	}

	d3d = Direct3DCreate9(D3D_SDK_VERSION);
	// Create a rendering context
	D3DPRESENT_PARAMETERS presentParams;
	ZeroMemory(&presentParams, sizeof(presentParams)); // Clear it out for our usage

	presentParams.Windowed = true;
	presentParams.SwapEffect = D3DSWAPEFFECT_DISCARD;
	presentParams.hDeviceWindow = window;

	d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING, &presentParams, &device);
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

	ImGui::StyleColorsDark();
	ImGui_ImplWin32_Init(window);
	ImGui_ImplDX9_Init(device);

	// Set our alive
	alive = true;

	renderer = std::thread(renderingFunc, device, &renderMutex);
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

	// Also, unregister the window class we created
	UnregisterClass(PUFFYPT_CLASS, GetModuleHandle(NULL));
	// The goal when closing, is to essentially act as if nothing ever happened, if we leave some type of trace (eg. memory leak or leftover resources),
	// then those could actually result in a crash if we open the module more than once

	printf("Removed window..\n");
	alive = false;
	FreeConsole();
	
	renderMutex.unlock();
}