#ifndef RENDER_H
#define RENDER_H

#include <d3d9.h>
#include <mutex>
#include <imgui.h>

extern void renderingFunc(LPDIRECT3DDEVICE9 device, std::mutex* renderMutex, ImFont* font);

#endif