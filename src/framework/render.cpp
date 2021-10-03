#include <framework/render.h>
#include <pathtracer/pathtracer.cuh>

#include <d3d9.h>
#include <mutex>
#include <memory>
#include <globals.h> // alive

#include <backends/imgui_impl_dx9.h>
#include <backends/imgui_impl_win32.h>
#include <imgui.h>

void renderingFunc(LPDIRECT3DDEVICE9 device, std::mutex* renderMutex, ImFont* font, std::shared_ptr<Pathtracer> pathtracer) {
	while (alive) {
		// Dont ask
		renderMutex->lock();

		// Double check since we may've just terminated
		if (alive == false) {
			// Terminate
			break;
		}

		device->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 40, 100), 1.0f, 0);
		device->BeginScene();

		// Start a new ImGui frame
		ImGui_ImplDX9_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Renderer");
		ImGui::PushFont(font);

		ImGui::Text("Yay!! (from Framework)");
		
		if (pathtracer != nullptr) {
			pathtracer->ImGuiUpdate();
		}
		else {
			ImGui::TextColored(ImVec4(1, 0, 0, 1), "The pathtracer was never initialized! This should never be reached!");
		}

		ImGui::PopFont();
		ImGui::End();
		
		// End it
		ImGui::EndFrame();
		device->SetRenderState(D3DRS_ZENABLE, FALSE);
		device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
		device->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);

		ImGui::Render();
		ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());

		device->EndScene();
		device->Present(NULL, NULL, NULL, NULL);

		renderMutex->unlock();
	}
}