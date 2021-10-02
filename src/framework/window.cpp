#include <framework/window.h>
#include <Windows.h>
#include <globals.h>

HWND gmodHWND;

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
	ptClass.lpfnWndProc = DefWindowProc;
	ptClass.hInstance = GetModuleHandle(NULL); // NULL returns the current process instance
	ptClass.lpszClassName = PUFFYPT_CLASS;
	ptClass.style = CS_HREDRAW | CS_VREDRAW;
	ptClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	ptClass.hbrBackground = (HBRUSH)COLOR_WINDOW;

	RegisterClass(&ptClass);
}

HWND CreatePuffyWindow() {
	InitializePuffyClass(); // Caller shouldn't have to call this themselves
	HWND ptWindow = CreateWindowEx(0, PUFFYPT_CLASS, PUFFYPT_WINDOW_NAME, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, NULL, NULL, GetModuleHandle(NULL), NULL);
	// None of this Win32 API code should never be touched again

	ShowWindow(ptWindow, SW_SHOW);
	return ptWindow;
}