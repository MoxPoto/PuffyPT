#include <dxhook/mainHook.h>
#include <d3d9.h>
#include <imgui_impl_dx9.h>
#include <imgui_impl_win32.h>
#include <Windows.h>
#include <vector>

namespace DXHook {
	// Updates the input things, why not use the normal backend? too cursed for me to go through--also this is
	// mainly for True Fullscreen, since ImGui will break on that because of cursed reasons I don't even know about..
	std::vector<int> keyCodes = {
		VK_BACK,
		VK_TAB,
		VK_RETURN,
		VK_SPACE,
		VK_LEFT,
		VK_UP,
		VK_RIGHT,
		VK_DOWN,
		VK_INSERT,
		VK_ADD,
		VK_MULTIPLY,
		VK_DIVIDE,
		VK_SUBTRACT,
		VK_DELETE // more are in here, just in mainDX.cpp 
	};

	void UpdateImGUI() {
		ImGuiIO& io = ImGui::GetIO();
		POINT currentMousePos;
		GetCursorPos(&currentMousePos);

		io.MousePos = ImVec2(currentMousePos.x, currentMousePos.y);

		if ((GetKeyState(VK_LBUTTON) & 0x8000) != 0) { // mouse down
			io.MouseDown[0] = true;
		}
		else {
			io.MouseDown[0] = false;
		}

		if ((GetKeyState(VK_RBUTTON) & 0x8000) != 0) { // mouse right down
			io.MouseDown[1] = true;
		}
		else {
			io.MouseDown[1] = false;
		} 

		if (io.WantCaptureMouse) { // do not ask
			for (int keyCode : keyCodes) {
				if ((GetKeyState(keyCode) & 0x8000) != 0) {
					io.KeysDown[keyCode] = true;
				}
				else {
					io.KeysDown[keyCode] = false;
				}
			}
		}
	}
}