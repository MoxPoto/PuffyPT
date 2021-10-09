#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <d3d9.h>
#include <Windows.h>
#include <thread>
#include <mutex>
#include <imgui.h>

#include <pathtracer/pathtracer.cuh>
#include <memory>

class Framework {
private:
	IDirect3DDevice9* device;
	IDirect3D9* d3d;
	D3DPRESENT_PARAMETERS presentParams;

	IDirect3DTexture9* renderTexture;
	ID3DXSprite* renderSprite;

	std::thread renderer;
	std::mutex renderMutex;

	HWND window;
	ImFont* font;

	std::shared_ptr<Pathtracer> pathtracer;
public:
	void InitWindow();
	void SetPathtracer(std::shared_ptr<Pathtracer> pathtracerPtr);

	Framework();
	~Framework();
};

#endif