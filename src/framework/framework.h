#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <d3d9.h>
#include <Windows.h>
#include <thread>
#include <mutex>
#include <imgui.h>

class Framework {
private:
	IDirect3DDevice9* device;
	IDirect3D9* d3d;
	D3DPRESENT_PARAMETERS presentParams;

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