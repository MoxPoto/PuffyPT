#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <d3dx9math.h>

#include "mainHook.h"
#include <d3d9.h>
#include <imgui_impl_dx9.h>
#include <imgui_impl_win32.h>
#include <Windows.h>
#include <iostream>
#include <string>
#include <chrono>

#include "../ray.cuh"
#include "../mesh.cuh"
#include "../vec3.cuh"
#include "../object.cuh"
#include "../triangle.cuh"
#include "../postprocess/mainDenoiser.cuh"
#include <chrono>
#include <random>

#include "../images/hdriUtility.cuh"


#define WIDTH 480
#define HEIGHT 270
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define HDRI_FOLDER "C:\\pathtracer\\hdrs"
#define PUFF_INCREMENT(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable += 0.1f; }
#define PUFF_DECREMENT(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable -= 0.1f; }

#define PUFF_INCREMENT_RESET(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable += 0.1f; frameCount = 0;}
#define PUFF_DECREMENT_RESET(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable -= 0.1f; frameCount = 0;}

#define VERSION "PUFFY PT - 0.08"

struct Vertex
{
	float _x, _y, _z;
	float _nx, _ny, _nz;
	float _u, _v; // texture coordinates
	static const DWORD FVF;
};

const DWORD Vertex::FVF = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1;

std::default_random_engine randEngine;
std::uniform_real_distribution<float> unif(0.0, 1.0);

namespace DXHook {
	EndScene oldFunc;
	void* d3d9Device[119];
	LPDIRECT3DDEVICE9 device;
	bool gotDevice = false;

	float* fb;
	float* postFB;
	Tracer::Camera mainCam;
	Tracer::SkyInfo skyInfo;

	Tracer::Object** world;
	curandState* d_rand_state;
	IDirect3DTexture9* pathtraceOutput = NULL;
	IDirect3DVertexBuffer9* quadVertexBuffer = NULL;
	ID3DXSprite* pathtraceObject = NULL;
	ID3DXFont* msgFont = NULL;
	float fov = 114.f;
	int currentPass = 2;
	Tracer::vec3 camDir;

	float azimuth[3] = { 1, 1, 1 };
	float zenith[3] = { 0.5f, 0.7f, 1.0f };

	Tracer::vec3* origin;
	float curX = 0, curY = 0, curZ = 0;
	float curPitch = 0, curYaw = 0, curRoll = 0;
	Tracer::Post::GBuffer* gbufferData;
	bool denoiserEnabled = true;
	bool showSky = true;
	int world_count = 0;
	int frameCount = 0;
	bool aabbOverride = false;

	Tracer::HDRI* mainHDRI = NULL;
	float* hdriData = NULL;

	int samples = 1;
	int max_depth = 6; // less than 4 results in really, really bad reflections
	bool showPathtracer = true;
	std::chrono::steady_clock::time_point lastTime;
	float curTime = 0.f;

	int curHDRI = 0;
	std::vector<std::string> hdriList;
	int hdriListSize = 0;
	float hdriBrightness = 1.f;

	HRESULT __stdcall EndSceneHook(LPDIRECT3DDEVICE9 pDevice) {
		using Tracer::vec3;

		if (!gotDevice) {
			gotDevice = true;
			device = pDevice;

			
			HRESULT failCode = D3DXCreateTexture(device, WIDTH, HEIGHT, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &pathtraceOutput);

			if (!pathtraceOutput) {
				
				std::cout << "Failed to create FB texture for the pathtracer.. Code: " << failCode << "\nD3DERR_INVALIDCALL: " << 
					D3DERR_INVALIDCALL << std::endl;
			}

			HRESULT failCode2 = D3DXCreateSprite(device, &pathtraceObject);

			if (!pathtraceObject) {

				std::cout << "Failed to create sprite for the pathtracer.. Code: " << failCode2 << "\nD3DERR_INVALIDCALL: " <<
					D3DERR_INVALIDCALL << std::endl;
			}


			HRESULT failCode3 = D3DXCreateFont(
				pDevice,
				18,
				0,
				FW_NORMAL,
				1,
				FALSE,
				DEFAULT_CHARSET,
				OUT_DEFAULT_PRECIS,
				ANTIALIASED_QUALITY,
				DEFAULT_PITCH | FF_DONTCARE,
				"Terminal",
				&msgFont
			);

			if (!msgFont) {

				std::cout << "Failed to create font for the pathtracer.. Code: " << failCode3 << "\nD3DERR_INVALIDCALL: " <<
					D3DERR_INVALIDCALL << std::endl;
			}

			pDevice->CreateVertexBuffer(6 * sizeof(Vertex), D3DUSAGE_WRITEONLY, Vertex::FVF, D3DPOOL_DEFAULT, &quadVertexBuffer, 0);

			Vertex* v;
			quadVertexBuffer->Lock(0, 0, (void**)&v, 0);

			// quad built from two triangles, note texture coordinates:
			v[0] = { -1.0f, -1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f }; // was Vertex()
			v[1] = { -1.0f, 1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f };
			v[2] = { 1.0f, 1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f };
			v[3] = { -1.0f, -1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f };
			v[4] = { 1.0f, 1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f };
			v[5] = { 1.0f, -1.0f, 1.25f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f };
			quadVertexBuffer->Unlock();

			ImGui_ImplDX9_Init(device);
		}


		HRESULT result = oldFunc(pDevice);

		D3DRASTER_STATUS rasterStatus;
		HRESULT statRes = pDevice->GetRasterStatus(0, &rasterStatus);

		ImGui_ImplWin32_NewFrame();
		ImGui_ImplDX9_NewFrame();

		ImGui::NewFrame();
		DXHook::UpdateImGUI();

		bool showWind = true;

		ImGui::SetNextWindowFocus();

	
		// test panel
		ImGui::Begin("Shader Modifier");

		ImGui::PushFont(ourFont);

		ImGui::Button("Increase FOV");

		if (ImGui::IsItemActive()) {
			fov += 1.f;
		};

		ImGui::Button("Decrease FOV");

		if (ImGui::IsItemActive()) {
			fov -= 1.f;
		}
		
		ImGui::Text("Current FOV: %.2f", fov);

		ImGui::Text("Current Samples: %d", samples);
		ImGui::Text("Current Max Depth: %d", max_depth);
		
		if (ImGui::Button("Increase Samples")) {
			samples += 5;
		}
		
		if (ImGui::Button("Decrease Samples")) {
			samples -= 5;
		}

		if (ImGui::Button("Increase Depth")) {
			max_depth += 1;
		}

		if (ImGui::Button("Decrease Depth")) {
			max_depth -= 1;
		}

		PUFF_INCREMENT("Exposure Increase", mainCam.exposure);
		PUFF_DECREMENT("Exposure Decrease", mainCam.exposure);
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "HDRI Index: %d", curHDRI);
		ImGui::Text("Current HDRI: ");
		ImGui::SameLine();
		ImGui::Text(hdriList.at(curHDRI).c_str());
		
		if (ImGui::Button("Left")) {
			curHDRI = max(min(curHDRI - 1, hdriListSize - 1), 0);
		
			Tracer::LoadHDRI(hdriList.at(curHDRI));
			frameCount = 0;
		}

		if (ImGui::Button("Right")) {
			curHDRI = max(min(curHDRI + 1, hdriListSize - 1), 0);

			Tracer::LoadHDRI(hdriList.at(curHDRI));
			frameCount = 0;
		}

		if (ImGui::Button("Refresh HDRI List")) {
			Tracer::FindHDRIs(HDRI_FOLDER, hdriList, hdriListSize);
		}

		PUFF_INCREMENT_RESET("Increase HDRI Brightness", hdriBrightness);
		PUFF_DECREMENT_RESET("Decrease HDRI Brightness", hdriBrightness);

		ImGui::Checkbox("Enable Postprocessing?", &denoiserEnabled);
		ImGui::Checkbox("Show Output?", &showPathtracer);
		ImGui::Checkbox("Show Sky?", &showSky);
		ImGui::Checkbox("Override AABB Accel?", &aabbOverride);

		if (ImGui::ColorPicker3("Edit Sky Azimuth", azimuth)) {
			frameCount = 0;
		}

		if (ImGui::ColorPicker3("Edit Sky Zenith", zenith)) {
			frameCount = 0;
		}

		skyInfo.azimuth = vec3(azimuth[0], azimuth[1], azimuth[2]);
		skyInfo.zenith = vec3(zenith[0], zenith[1], zenith[2]);

		const char* passes[] = {
			"Direct Lighting",
			"Indirect Lighting",
			"Combined"
		};

		if (ImGui::ListBox("Passes", &currentPass, passes, 3))
			frameCount = 0;

		ImGui::End();

		ImGui::PopFont();

		ImGui::EndFrame();

		curTime = unif(randEngine); // hel p


		int warpX = 16;
		int warpY = 16; // technically can be ruled out as tiled rendering

		dim3 blocks(WIDTH / warpX + 1, HEIGHT / warpY + 1);
		dim3 threads(warpX, warpY);


		// std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

		if (showPathtracer) {
			RenderOptions options;
			options.count = world_count;
			options.fov = fov;
			options.x = curX;
			options.y = curY;
			options.z = curZ;
			options.pitch = curPitch;
			options.yaw = curYaw;
			options.roll = curRoll;
			options.frameBuffer = fb;
			options.world = world;
			options.max_x = WIDTH;
			options.max_y = HEIGHT;
			options.rand_state = d_rand_state;
			options.samples = samples;
			options.max_depth = max_depth;
			options.gbufferPtr = gbufferData;
			options.frameCount = frameCount;
			options.curtime = curTime;
			options.doSky = showSky;
			options.hdri = mainHDRI;
			options.hdriData = hdriData;
			options.curPass = currentPass;
			options.skyInfo = skyInfo;
			options.cameraDir = camDir;
			options.aabbOverride = aabbOverride;
			options.hdriBrightness = hdriBrightness;
			render << <blocks, threads >> > (options);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			frameCount++;

			if (denoiserEnabled) {
				Tracer::ApplyPostprocess(WIDTH, HEIGHT, blocks, threads);
			}

		}
		// std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		// double timeSpent = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		// std::cout << "Finished rendering in " << timeSpent << " milliseconds, saving to tex\n";

		if (pathtraceOutput) {
			D3DLOCKED_RECT memRegion;
			pathtraceOutput->LockRect(0, &memRegion, NULL, D3DLOCK_DISCARD);
			// std::cout << "Updating texture pt 1\n";
			int num_pixels = WIDTH * HEIGHT;
			
			unsigned char* dest = static_cast<unsigned char*>(memRegion.pBits);
			//unsigned char* newFb = static_cast<unsigned char*>(malloc(4 * num_pixels * sizeof(unsigned char)));
			// SO GET THIS I WAS FUCKING MALLOC'ING THIS EVERY FRAME AND FORGETTING TO REMOVE IT
			// AND MY SYSTEM BSODED AND MY GPU GOT STUCK IN A RANDOM STATE LMAOOOOOO

			char* data = reinterpret_cast<char*>(memRegion.pBits);

			for (int y = 0; y < HEIGHT; ++y) {
				DWORD* row = (DWORD*)data;
				for (int x = 0; x < WIDTH; ++x) {
					int pixel_index = y * WIDTH * 3 + x * 3;
					int r = int(postFB[pixel_index] * 255.99);
					int g = int(postFB[pixel_index + 1] * 255.99);
					int b = int(postFB[pixel_index + 2] * 255.99);

					if (!denoiserEnabled) {
						r = int(fb[pixel_index] * 255.99);
						g = int(fb[pixel_index + 1] * 255.99);
						b = int(fb[pixel_index + 2] * 255.99);
					}

					*row++ = D3DCOLOR_XRGB(r, g, b);
				}
				data += memRegion.Pitch;
			}

			pathtraceOutput->UnlockRect(0);

			if (pathtraceObject) {
				D3DXMATRIX transformation;

				D3DXMatrixIdentity(&transformation);
				D3DXMatrixScaling(&transformation, 4, 4, 1);

				if (showPathtracer) {
					pathtraceObject->Begin(D3DXSPRITE_SORT_DEPTH_FRONTTOBACK);
					pathtraceObject->SetTransform(&transformation);
					pathtraceObject->Draw(pathtraceOutput, NULL, NULL, &D3DXVECTOR3(0.3, 0.3, 1), D3DCOLOR_RGBA(255, 255, 255, 255));
					pathtraceObject->End();
				}

				if (msgFont) {
					RECT msgRect;
					SetRect(&msgRect, 0, 15, 1920, 120);

					msgFont->DrawText(NULL, VERSION, -1, &msgRect, DT_CENTER | DT_NOCLIP, D3DCOLOR_ARGB(255, 10, 10, 10));


				}

				pDevice->SetRenderState(D3DRS_ZENABLE, FALSE);
				pDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
				pDevice->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);

				ImGui::Render();
				ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
			}

		}

		return result;
	}


}