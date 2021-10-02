#ifndef FRAMEWORK_H
#define FRAMEWORK_H

#include <d3d9.h>
#include <Windows.h>
#include <thread>
#include <mutex>

class Framework {
private:
	IDirect3DDevice9* device;
	IDirect3D9* d3d;
	std::thread renderer;
	std::mutex* renderMutex;
	HWND window;

	bool alive;
public:
	void InitWindow();

	Framework();
	~Framework();
};

#endif