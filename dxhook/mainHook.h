#ifndef MAINHOOK_H
#define MAINHOOK_H

#include <d3d9.h>
#include <d3dx9.h>
#include "GarrysMod/Lua/Interface.h"
#include <imgui.h>
#include <Windows.h>
#include <vector>

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "../mesh.cuh"
#include "../denoiser/mainDenoiser.cuh"

namespace DXHook {
	typedef HRESULT(__stdcall* EndScene)(LPDIRECT3DDEVICE9);

	extern EndScene oldFunc;
	extern void* d3d9Device[119];
	extern HRESULT __stdcall EndSceneHook(LPDIRECT3DDEVICE9 pDevice);

	extern float* fb; // Frame buffer
	extern Tracer::Object** world;
	extern curandState* d_rand_state;
	extern IDirect3DTexture9* pathtraceOutput;
	extern IDirect3DVertexBuffer9* quadVertexBuffer;
	extern ID3DXSprite* pathtraceObject;

	extern float fov;
	extern Tracer::vec3* origin;
	extern float curX, curY, curZ;
	extern float curPitch, curYaw, curRoll;
	extern ID3DXFont* msgFont;
	extern Tracer::Denoising::GBuffer** gbufferData;

	extern HWND window;
	extern bool gotDevice;
	extern LPDIRECT3DDEVICE9 device;
	extern LONG_PTR originalWNDPROC;

	extern BOOL CALLBACK EnumWindowsCallback(HWND handle, LPARAM lParam);
	extern HWND GetProcessWindow();
	extern bool GetD3D9Device(void** pTable, size_t Size);
	extern std::vector<int> keyCodes;
	extern ImFont* ourFont;
	extern int samples, max_depth;

	struct RenderOptions {
		float* frameBuffer;
		Tracer::Object** world;
		float x;
		float y;
		float z; 
		float pitch;
		float yaw;
		float roll;
		curandState* rand_state;
		int count;
		float fov;
		int max_x;
		int max_y;
		int samples;
		int max_depth;
		Tracer::Denoising::GBuffer** gbufferPtr;
	};

	extern __global__ void render(RenderOptions options);
	extern __global__ void initMem(Tracer::Object** world, Tracer::vec3* origin);
	extern __global__ void registerRands(int max_x, int max_y, curandState* rand_state, Tracer::Denoising::GBuffer** gbufferData);
	extern void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

	extern inline void error(GarrysMod::Lua::ILuaBase* LUA, const char* str);
	extern int Initialize(GarrysMod::Lua::ILuaBase* LUA); // Used for setting up dummy device, and endscene hook
	extern int Cleanup(GarrysMod::Lua::ILuaBase* LUA); // Used for restoring the EndScene
	extern void UpdateImGUI();
}

#endif 