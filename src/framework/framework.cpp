#include <framework/framework.h>
#include <framework/window.h>

#include <d3d9.h>
#include <Windows.h>
#include <stdio.h>

// There should be some way to make this.. better
static void renderingFunc(LPDIRECT3DDEVICE9 device, std::mutex* renderMutex, bool* alive) {
	while (*alive) {
		// Dont ask
		renderMutex->lock();

		// Double check since we may've just terminated
		if (*alive == false) {
			// Terminate
			renderMutex->unlock();
			break;
		}

		device->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 40, 100), 1.0f, 0);
		device->BeginScene();

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

	// Set our alive
	alive = true;

	renderMutex = new std::mutex();
	renderMutex->unlock();

	renderer = std::thread(renderingFunc, device, renderMutex, &alive);
	renderer.detach();
}

Framework::~Framework() {
	renderMutex->lock(); // Lock real quick

	// Close the main window
	// And free the console
	FreeConsole();

	// Cleanup D3D resources
	device->Release();
	d3d->Release();

	if (window != NULL) {
		DestroyWindow(window);
	}

	alive = false;
	
	renderMutex->unlock();

	delete renderMutex;
}