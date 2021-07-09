#include "mainHook.h"
#include <d3d9.h>
#include <Windows.h>

namespace DXHook {
	HWND window;

	BOOL CALLBACK EnumWindowsCallback(HWND handle, LPARAM lParam) {
		DWORD wndProcId;
		GetWindowThreadProcessId(handle, &wndProcId);

		if (GetCurrentProcessId() != wndProcId)
			return TRUE; // skip to next window

		window = handle;
		return FALSE; // window found abort search
	}

	HWND GetProcessWindow() {
		window = NULL;
		EnumWindows(EnumWindowsCallback, NULL);
		return window;
	}
}