#ifndef WINDOW_H
#define WINDOW_H
// Helper for creating the Puffy PT window class

#include <Windows.h>

const LPCSTR PUFFYPT_CLASS = "PuffyPTWindow";
extern HWND gmodHWND;

// Enumerate through every window and check if its the GMod window
extern BOOL CALLBACK FindGModWindow(HWND handle, LPARAM lParam);
// Initialize the Puffy PT Window class
extern void InitializePuffyClass();
// Create the Puffy PT Window
extern HWND CreatePuffyWindow();
// Sets gmodHWND to the GMod window
// CHECK FOR NULL RETURNS!!
extern HWND GetGModWindow();

#endif