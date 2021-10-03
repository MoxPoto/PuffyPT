#ifndef RENDER_H
#define RENDER_H

#include <pathtracer/pathtracer.cuh>

#include <d3d9.h>
#include <mutex>
#include <imgui.h>
#include <memory>

extern void renderingFunc(LPDIRECT3DDEVICE9 device, std::mutex* renderMutex, ImFont* font, std::shared_ptr<Pathtracer> pathtracer);

#endif