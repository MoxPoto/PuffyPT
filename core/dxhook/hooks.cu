#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include <d3dx9math.h>

#include <dxhook/mainHook.h>
#include <d3d9.h>
#include <imgui_impl_dx9.h>
#include <imgui_impl_win32.h>
#include <Windows.h>
#include <iostream>
#include <string>
#include <chrono>

#include <classes/ray.cuh>
#include <classes/mesh.cuh>
#include <classes/vec3.cuh>
#include <classes/object.cuh>
#include <classes/triangle.cuh>
#include <postprocess/mainDenoiser.cuh>

#include <chrono>
#include <random>
#include <mutex>

#include <images/hdriUtility.cuh>

#define WIDTH 960
#define HEIGHT 540
#define checkCudaErrors(val) DXHook::check_cuda( (val), #val, __FILE__, __LINE__ )
#define HDRI_FOLDER "C:\\pathtracer\\hdrs"
#define PUFF_INCREMENT(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable += 0.1f; }
#define PUFF_DECREMENT(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable -= 0.1f; }

#define PUFF_INCREMENT_RESET(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable += 0.1f; frameCount = 0; renderOptDevPtr->hdriBrightness = hdriBrightness;}
#define PUFF_DECREMENT_RESET(name, variable) ImGui::Button(name); if (ImGui::IsItemActive()) { variable -= 0.1f; frameCount = 0; renderOptDevPtr->hdriBrightness = hdriBrightness;}

#define VERSION "PUFFY PT - 2.3"

// note to self:
// CUDA's compiler (nvcc) is shit (or me using it)
// for the past 8 hours, most of the features in my tracer wouldnt work
// i couldnt understand why at all, looked perfect
// reason?
// cuda wasn't compiling the new object files
// rebuild all and it works perfectly.. sigh

std::default_random_engine randEngine;
std::uniform_real_distribution<float> unif(0.0, 1.0);

static int accumulatedSamples = 0;

#define MEASURE_START(id) std::chrono::steady_clock::time_point id = std::chrono::high_resolution_clock::now();
#define MEASURE_END(id, result) double result = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - id).count();

namespace DXHook {
	EndScene oldFunc;
	void* d3d9Device[119];
	LPDIRECT3DDEVICE9 device;
	bool gotDevice = false;

	float* fb;
	float* postFB;
	float* bloomFB;
	float* blurFB;
	Camera mainCam;
	SkyInfo skyInfo;

	Object** world;
	curandState* d_rand_state;
	IDirect3DTexture9* pathtraceOutput = NULL;
	ID3DXSprite* pathtraceObject = NULL;
	ID3DXFont* msgFont = NULL;
	float fov = 114.f;
	int currentPass = 2;
	vec3 camDir;

	float azimuth[3] = { 1, 1, 1 };
	float zenith[3] = { 0.5f, 0.7f, 1.0f };

	float whiteBalance = 5706.f;

	vec3* origin;
	float curX = 0, curY = 0, curZ = 0;
	float curPitch = 0, curYaw = 0, curRoll = 0;
	Post::GBuffer* gbufferData;
	bool denoiserEnabled = true;
	bool showSky = true;
	int world_count = 0;
	int frameCount = 0;
	bool aabbOverride = false;

	HDRI* mainHDRI = NULL;
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
	RendererType curRender = 0;

	bool denoiseImage = true;
	bool cleanedUp = false;

	RenderOptions* renderOptDevPtr = nullptr;
	std::mutex* renderMutex;

	// DX Framebuffer
	DWORD* dxFB = nullptr;

	HRESULT __stdcall EndSceneHook(LPDIRECT3DDEVICE9 pDevice) {
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
				36,
				0,
				FW_NORMAL,
				1,
				FALSE,
				DEFAULT_CHARSET,
				OUT_DEFAULT_PRECIS,
				ANTIALIASED_QUALITY,
				DEFAULT_PITCH | FF_DONTCARE,
				"CoolveticaRg-Regular",
				&msgFont
			);

			if (!msgFont) {

				std::cout << "Failed to create font for the pathtracer.. Code: " << failCode3 << "\nD3DERR_INVALIDCALL: " <<
					D3DERR_INVALIDCALL << std::endl;
			}

			ImGui_ImplDX9_Init(device);
		}

		renderMutex->lock();

		HRESULT result = oldFunc(pDevice);

		if (DXHook::cleanedUp) {
			return result;
		}

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
			frameCount = 0;
		};

		ImGui::Button("Decrease FOV");

		if (ImGui::IsItemActive()) {
			fov -= 1.f;
			frameCount = 0;
		}
		
		ImGui::Text("Current FOV: %.2f", fov);

		if (frameCount <= 0) {
			accumulatedSamples = 0;
		}

		accumulatedSamples += samples;

		ImGui::Text("Accumulated Samples: %d", accumulatedSamples);
		ImGui::Text("Current Samples: %d", samples);
		ImGui::Text("Current Max Depth: %d", max_depth);
		
		if (ImGui::Button("Increase Samples")) {
			samples += 1;

			renderOptDevPtr->samples = samples;
		}
		
		if (ImGui::Button("Decrease Samples")) {
			samples -= 1;

			renderOptDevPtr->samples = samples;
		}

		if (ImGui::Button("Increase Depth")) {
			max_depth += 1;
			renderOptDevPtr->max_depth = max_depth;
		}

		if (ImGui::Button("Decrease Depth")) {
			max_depth -= 1;
			renderOptDevPtr->max_depth = max_depth;
		}

		ImGui::SliderFloat("White Balance", &whiteBalance, 1668.f, 24999.f, "%.3f");
		// ImGui::SliderFloat("HDRI Pitch", &renderOptDevPtr->hdriPitch, -360, 360, "%.2f");
		
		if (ImGui::SliderFloat("HDRI Yaw", &renderOptDevPtr->hdriYaw, -360, 360, "%.2f")) {
			frameCount = 0;
		}

		PUFF_INCREMENT("Exposure Increase", mainCam.exposure);
		PUFF_DECREMENT("Exposure Decrease", mainCam.exposure);
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "HDRI Index: %d", curHDRI);
		ImGui::Text("Current HDRI: ");
		ImGui::SameLine();
		ImGui::Text(hdriList.at(curHDRI).c_str());
		
		if (ImGui::Button("Left")) {
			curHDRI = max(min(curHDRI - 1, hdriListSize - 1), 0);
		
			LoadHDRI(hdriList.at(curHDRI));
			frameCount = 0;
		}

		if (ImGui::Button("Right")) {
			curHDRI = max(min(curHDRI + 1, hdriListSize - 1), 0);

			LoadHDRI(hdriList.at(curHDRI));
			frameCount = 0;
		}

		if (ImGui::Button("Refresh HDRI List")) {
			FindHDRIs(HDRI_FOLDER, hdriList, hdriListSize);
		}

		PUFF_INCREMENT_RESET("Increase HDRI Brightness", hdriBrightness);
		PUFF_DECREMENT_RESET("Decrease HDRI Brightness", hdriBrightness);

		char imageName[100];

		ImGui::Text("Type render name to save the render to disk");

		if (ImGui::InputText("##Render Name", imageName, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
			D3DXSaveTextureToFile((std::string("C:\\pathtracer\\") + std::string(imageName) + std::string(".png")).c_str(), D3DXIFF_PNG, pathtraceOutput, NULL);
		};
		

		ImGui::Checkbox("Denoise Image?", &denoiseImage);
		ImGui::Checkbox("Enable Postprocessing?", &denoiserEnabled);
		ImGui::Checkbox("Show Output?", &showPathtracer);
		if (ImGui::Checkbox("Show Sky?", &showSky)) {
			renderOptDevPtr->doSky = showSky;
		}

		// ImGui::Checkbox("Override AABB Accel?", &aabbOverride);

		/*
		if (ImGui::ColorPicker3("Edit Sky Azimuth", azimuth)) {
			frameCount = 0;
		}

		if (ImGui::ColorPicker3("Edit Sky Zenith", zenith)) {
			frameCount = 0;
		}

		skyInfo.azimuth = vec3(azimuth[0], azimuth[1], azimuth[2]);
		skyInfo.zenith = vec3(zenith[0], zenith[1], zenith[2]);
		*/

		// No practical use for these ^

		const char* passes[] = {
			"Direct Lighting",
			"Indirect Lighting",
			"Combined"
		};

		const char* renderers[] = {
			"Puffy PT",
			"Puffy MLT",
			"Puffy Simple RT"
		};

		if (ImGui::ListBox("Renderers", &curRender, renderers, 3)) {
			frameCount = 0;
			renderOptDevPtr->renderer = curRender;
		}

		if (ImGui::ListBox("Passes", &currentPass, passes, 3)) {
			frameCount = 0;
			renderOptDevPtr->curPass = currentPass;
		}

		ImGui::End();

		ImGui::PopFont();

		ImGui::EndFrame();

		curTime = unif(randEngine); // hel p


		int warpX = 6;
		int warpY = 6; // technically can be ruled out as tiled rendering

		dim3 blocks(WIDTH / warpX + 1, HEIGHT / warpY + 1);
		dim3 threads(warpX, warpY);


		// std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();

		if (showPathtracer) {
			MEASURE_START(pathtraceTime);
			renderOptDevPtr->count = world_count;

			// Abuse the fact that the framecount is reset everytime the user moves
			// 10 to make sure (aka, padding)
			if (frameCount < 10) {
				renderOptDevPtr->x = curX;
				renderOptDevPtr->y = curY;
				renderOptDevPtr->z = curZ;
				renderOptDevPtr->pitch = curPitch;
				renderOptDevPtr->yaw = curYaw;
				renderOptDevPtr->roll = curRoll;

				renderOptDevPtr->fov = fov;

				renderOptDevPtr->cameraDir = camDir;
			}

			renderOptDevPtr->frameCount = frameCount;
			renderOptDevPtr->curtime = curTime;

			// printf("test: %d\n", renderOptDevPtr->max_x);
			
			render << <blocks, threads >> > (renderOptDevPtr);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());

			frameCount++;

			if (denoiserEnabled) {
				ApplyPostprocess(WIDTH, HEIGHT, blocks, threads, denoiseImage, whiteBalance);
			}

			MEASURE_END(pathtraceTime, pathtraceTimeDouble);

			// printf("[host]: Time took to render in ms: %.4f\n", pathtraceTimeDouble);

		}
		// std::chrono::steady_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		// double timeSpent = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		// std::cout << "Finished rendering in " << timeSpent << " milliseconds, saving to tex\n";

		if (pathtraceOutput) {
			MEASURE_START(texTransfer);

			D3DLOCKED_RECT memRegion;
			pathtraceOutput->LockRect(0, &memRegion, NULL, 0);
			// std::cout << "Updating texture pt 1\n";
			int num_pixels = WIDTH * HEIGHT;
			
			unsigned char* dest = static_cast<unsigned char*>(memRegion.pBits);
			//unsigned char* newFb = static_cast<unsigned char*>(malloc(4 * num_pixels * sizeof(unsigned char)));
			// SO GET THIS I WAS FUCKING MALLOC'ING THIS EVERY FRAME AND FORGETTING TO REMOVE IT
			// AND MY SYSTEM BSODED AND MY GPU GOT STUCK IN A RANDOM STATE LMAOOOOOO

			DWORD* data = reinterpret_cast<DWORD*>(memRegion.pBits);
			/*
			for (int y = 0; y < HEIGHT; ++y) {
				DWORD* row = (DWORD*)data;

				for (int x = 0; x < WIDTH; ++x) {
					int pixel_index = y * WIDTH * 3 + x * 3;
					int r = static_cast<int>(postFB[pixel_index] * 255.99);
					int g = static_cast<int>(postFB[pixel_index + 1] * 255.99);
					int b = static_cast<int>(postFB[pixel_index + 2] * 255.99);

					if (!denoiserEnabled) {
						r = static_cast<int>(fb[pixel_index] * 255.99);
						g = static_cast<int>(fb[pixel_index + 1] * 255.99);
						b = static_cast<int>(fb[pixel_index + 2] * 255.99);
					}

					*row++ = D3DCOLOR_XRGB(r, g, b);
				}
				data += memRegion.Pitch;
			}
			*/

			memcpy(data, (void*)DXHook::dxFB, WIDTH * HEIGHT * sizeof(DWORD));

			pathtraceOutput->UnlockRect(0);

			MEASURE_END(texTransfer, texTransferTime);
			// printf("[host]: Time took to transfer render in ms: %.4f\n", texTransferTime);

			if (pathtraceObject) {
				D3DXMATRIX transformation;

				D3DXMatrixIdentity(&transformation);
				D3DXMatrixScaling(&transformation, 1920 / WIDTH, 1080 / HEIGHT, 1);

				if (showPathtracer) {
					pathtraceObject->Begin(D3DXSPRITE_SORT_DEPTH_FRONTTOBACK);
					pathtraceObject->SetTransform(&transformation);
					pathtraceObject->Draw(pathtraceOutput, NULL, NULL, &D3DXVECTOR3(0.3, 0.3, 1), D3DCOLOR_RGBA(255, 255, 255, 255));
					pathtraceObject->End();
				}

				if (msgFont) {
					RECT msgRect;
					SetRect(&msgRect, 0, 1010, 1920, 120);

					msgFont->DrawText(NULL, VERSION, -1, &msgRect, DT_CENTER | DT_NOCLIP, D3DCOLOR_ARGB(90, 255, 255, 255));


				}

				pDevice->SetRenderState(D3DRS_ZENABLE, FALSE);
				pDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
				pDevice->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);

				ImGui::Render();
				ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
			}

		}

		renderMutex->unlock();

		return result;
	}


}