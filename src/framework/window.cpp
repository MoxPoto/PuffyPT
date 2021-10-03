#include <framework/window.h>
#include <Windows.h>
#include <globals.h>

#include <imgui.h>
#include <backends/imgui_impl_win32.h>

HWND gmodHWND;

// Get the WndProc handler in here
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	return DefWindowProc(hWnd, msg, wParam, lParam);
}

BOOL CALLBACK FindGModWindow(HWND handle, LPARAM lParam) {
	DWORD processId;
	GetWindowThreadProcessId(handle, &processId);

	if (GetCurrentProcessId() != processId)
		return TRUE; // This window isn't it..

	// if we found it, then save and abort
	gmodHWND = handle;
	return FALSE; // Abort
}

HWND GetGModWindow() {
	EnumWindows(FindGModWindow, NULL);
	return gmodHWND;
}

void InitializePuffyClass() {
	WNDCLASS ptClass = {};
	ptClass.lpfnWndProc = WndProc;
	ptClass.hInstance = GetModuleHandle(NULL); // NULL returns the current process instance
	ptClass.lpszClassName = PUFFYPT_CLASS;
	ptClass.style = CS_HREDRAW | CS_VREDRAW;
	ptClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	ptClass.hbrBackground = (HBRUSH)COLOR_WINDOW;

	RegisterClass(&ptClass);
}

HWND CreatePuffyWindow() {
	InitializePuffyClass(); // Caller shouldn't have to call this themselves
	HWND ptWindow = CreateWindowEx(0, PUFFYPT_CLASS, PUFFYPT_WINDOW_NAME, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1728, 972, NULL, NULL, GetModuleHandle(NULL), NULL);
	// None of this Win32 API code should never be touched again

	ShowWindow(ptWindow, SW_SHOW);
	return ptWindow;
}