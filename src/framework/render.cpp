#include <framework/render.h>

#include <d3d11.h>
#include <d3d10.h>

#include <mutex>
#include <memory>
#include <globals.h> // alive

#include <backends/imgui_impl_dx11.h>
#include <backends/imgui_impl_win32.h>
#include <imgui.h>

#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

void renderingFunc(ComPtr<ID3D11Device> device, ComPtr<IDXGISwapChain> swapChain, ComPtr<ID3D11DeviceContext> devContext, ComPtr<ID3D11RenderTargetView> backbuffer, std::mutex* renderMutex, ImFont* font) {
	while (alive) {
		// Dont ask
		renderMutex->lock();

		// Double check since we may've just terminated
		if (alive == false) {
			// Terminate
			break;
		}

		float newColor[4] = { 0.0f, 0.2f, 0.4f, 1.0f };

		devContext->ClearRenderTargetView(backbuffer.Get(), newColor);

		// Start a new ImGui frame
		ImGui_ImplDX11_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Puffy PT");
		ImGui::PushFont(font);
		
		ImGui::Text("FPS: %.2f", ImGui::GetIO().Framerate);

		if (ImGui::Button("Recompile Puffy PT")) {
			framework->CompilePuffyPT();
		}

		ImGui::PopFont();
		ImGui::End();
		
		// End it
		ImGui::EndFrame();

		ImGui::Render();
		ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

		swapChain->Present(0, 0);

		renderMutex->unlock();
	}
}