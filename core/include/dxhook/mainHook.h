#ifndef MAINHOOK_H
#define MAINHOOK_H

#include <d3d9.h>
#include <d3dx9.h>
#include "GarrysMod/Lua/Interface.h"
#include <imgui.h>
#include <Windows.h>
#include <vector>
#include <chrono>
#include <string>

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <classes/mesh.cuh>
#include <postprocess/mainDenoiser.cuh>
#include <images/hdri.cuh>
#include <classes/camera.cuh>

#include <mutex>

namespace DXHook {
	typedef HRESULT(__stdcall* EndScene)(LPDIRECT3DDEVICE9);
	typedef int RendererType;

	enum RendererTypes {
		PuffyPT,
		PuffyMLT,
		PuffySimpleRT
	};

	extern EndScene oldFunc;
	extern void* d3d9Device[119];
	extern HRESULT __stdcall EndSceneHook(LPDIRECT3DDEVICE9 pDevice);

	extern float* fb; // Frame buffer
	extern float* postFB; // Post Frame buffer
	extern float* bloomFB; // Bloom Frame buffer
	extern float* blurFB; // Pre-blur FB
	extern DWORD* dxFB; // DirectX Framebuffer

	extern float whiteBalance; // White Balance Dumbass

	extern Camera mainCam; // Main camera
	extern SkyInfo skyInfo;

	extern Object** world;
	extern curandState* d_rand_state;
	extern IDirect3DTexture9* pathtraceOutput;
	extern ID3DXSprite* pathtraceObject;
	extern HDRI* mainHDRI;
	extern float* hdriData;
	extern int currentPass;
	extern int curHDRI;
	extern std::vector<std::string> hdriList;
	extern int hdriListSize;
	
	extern bool denoiseImage; // denoise image or not

	extern float fov;
	extern vec3* origin;
	extern float curX, curY, curZ;
	extern float curPitch, curYaw, curRoll;
	extern ID3DXFont* msgFont;
	extern Post::GBuffer* gbufferData;
	extern bool denoiserEnabled;
	extern int world_count;
	extern bool showPathtracer;
	extern int frameCount;
	extern std::chrono::steady_clock::time_point lastTime;
	extern float curTime;
	extern bool showSky;
	extern bool aabbOverride;
	extern vec3 camDir;

	extern float zenith[3]; // for imgui
	extern float azimuth[3]; // for imgui

	extern RendererType curRender;

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

	extern float hdriBrightness;
	extern std::mutex* renderMutex;

	struct RenderOptions {
		float* frameBuffer;
		Object** world;
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
		int frameCount;
		float curtime;
		bool doSky;
		bool aabbOverride;
		vec3 cameraDir;
		int curPass;
		HDRI* hdri;
		float* hdriData;
		SkyInfo skyInfo;
		Post::GBuffer* gbufferPtr;
		float hdriBrightness; // This is just to communicate wanted HDRI brightness from host to device quicker
		RendererType renderer;

		float hdriPitch;
		float hdriYaw;
	};

	extern RenderOptions* renderOptDevPtr;
	extern bool cleanedUp;

	extern __global__ void render(RenderOptions* options);
	extern __global__ void initMem(Object** world, vec3* origin);
	extern __global__ void registerRands(int max_x, int max_y, curandState* rand_state, Post::GBuffer* gbufferData);
	extern void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

	extern inline void error(GarrysMod::Lua::ILuaBase* LUA, const char* str);
	extern int Initialize(GarrysMod::Lua::ILuaBase* LUA); // Used for setting up dummy device, and endscene hook
	extern int Cleanup(GarrysMod::Lua::ILuaBase* LUA); // Used for restoring the EndScene
	extern void UpdateImGUI();
}

#endif 