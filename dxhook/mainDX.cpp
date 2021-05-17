#include "GarrysMod/Lua/Interface.h"
#include "mainHook.h"
#include <Windows.h>
#include <iostream>
#include <sstream>
#include "../detours.h"


#include <imgui.h>
#include <imgui_impl_dx9.h>
#include <imgui_impl_win32.h>

// new window procedure payload
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    /* Currently need a way to delegate input to ImGui whilest handling it accordingly to gmod aswell..
    if (ImGui::GetIO().WantCaptureKeyboard || ImGui::GetIO().WantCaptureMouse)
        return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
    */

    ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
    ImGuiIO& io = ImGui::GetIO();

    if (io.WantCaptureKeyboard && io.WantCaptureMouse) {
        DefWindowProc(hWnd, msg, wParam, lParam);

        return true; // block the input from reaching the gmod window
    }

    return CallWindowProc((WNDPROC)DXHook::originalWNDPROC, hWnd, msg, wParam, lParam);
}

namespace DXHook {
    LONG_PTR originalWNDPROC;
    ImFont* ourFont;

	inline void error(GarrysMod::Lua::ILuaBase* LUA, const char* str) {
		LUA->PushSpecial(GarrysMod::Lua::SPECIAL_GLOB);
		LUA->GetField(-1, "print");
		LUA->PushString(str);
        
		LUA->Call(1, 0);
		LUA->Pop();
	}

	int Initialize(GarrysMod::Lua::ILuaBase* LUA) { // Used for setting up dummy device, and endscene hook
        HMODULE hDLL;
        hDLL = GetModuleHandleA("d3d9.dll"); // Attempt to locate the d3d9 dll that gmod loaded

        if (hDLL == NULL) {
            error(LUA, "Unable to locate d3d9.dll in the loaded memory?");
            return 0;
        }


        if (GetD3D9Device(d3d9Device, sizeof(d3d9Device)))
        {
            //hook stuff using the dumped addresses


            std::stringstream hexLoc;
            hexLoc.setf(std::ios_base::hex, std::ios_base::basefield);

            hexLoc << std::string("EndScene functions address: ") << (uintptr_t)d3d9Device[42];

            error(LUA, hexLoc.str().c_str()); // it's called "error" but its more of a print

            oldFunc = (EndScene)d3d9Device[42]; // cast our void* address to a function description btw 42 is the EndScene index

            DetourTransactionBegin(); // use MS detours to detour our EndScene 
            DetourUpdateThread(GetCurrentThread());
            DetourAttach(&(PVOID&)oldFunc, EndSceneHook); // if im gon be straight honest i took this from guidedhacking and I don't entirely get
            // the "&(PVOID&)"

            LONG lError = DetourTransactionCommit(); // execute it
            if (lError != NO_ERROR) {
                MessageBox(HWND_DESKTOP, "failed to detour", "puffy", MB_OK);
                return FALSE;
            }
            else {
                error(LUA, "Successfully got EndScene address, converted to function--detoured and ready");
            }

        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();// (void)io;
        
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        RECT resolutionDetails;
        GetClientRect(GetProcessWindow(), &resolutionDetails);

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        ImGui_ImplWin32_Init(GetProcessWindow());

        ImGui::GetMainViewport()->WorkSize = ImVec2(1920, 1080);
        io.DisplaySize.x = 1920; // width
        io.DisplaySize.y = 1080; // height
        ImGuiStyle& style = ImGui::GetStyle();

        style.FrameRounding = 5.0f;
        style.ChildRounding = 5.0f;
        style.WindowRounding = 5.0f;
        style.WindowTitleAlign = ImVec2(0.5, 0.5);
        style.Colors[ImGuiCol_TitleBg] = ImVec4(0.592, 0.090, 0.090, 1);
        style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.392, 0.020, 0.020, 1);

        ourFont = io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\micross.ttf", 15);

        // setup key codes

        // 0-9 key codes
        for (int i = 0x30; i <= 0x39; i++) {
            keyCodes.push_back(i);
        }

        // A-Z key codes
        for (int i = 0x41; i <= 0x5A; i++) {
            keyCodes.push_back(i);
        }

        // cursed C++ is what i live for
        LONG_PTR currentWndProc = GetWindowLongPtr(GetProcessWindow(), GWLP_WNDPROC);
        originalWNDPROC = currentWndProc;

        SetWindowLongPtr(GetProcessWindow(), GWLP_WNDPROC, (LONG_PTR)&WndProc);

        Sleep(150);



        return 0;
	} 

    int Cleanup(GarrysMod::Lua::ILuaBase* LUA) {
        FreeConsole();

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourDetach(&(PVOID&)oldFunc, EndSceneHook);

        LONG lError = DetourTransactionCommit();
        if (lError != NO_ERROR) {
            MessageBox(HWND_DESKTOP, "failed to revert detour", "puffy", MB_OK);
            return FALSE;
        }
        else {
            error(LUA, "Reverted detour on EndScene..");

            ImGui_ImplDX9_Shutdown();
            ImGui_ImplWin32_Shutdown();
            ImGui::DestroyContext();

            error(LUA, "Restoring original window procedure for gmod..");

            SetWindowLongPtr(GetProcessWindow(), GWLP_WNDPROC, originalWNDPROC);
        }

        return 0;
    } // Used for restoring the EndScene
}