#include "mainHook.h"
#include <d3d9.h>

namespace DXHook {
    // i forgot who made this but it wasnt me but atleast I understand it
	bool GetD3D9Device(void** pTable, size_t Size) {
        if (!pTable) // make sure we have a valid destination of an array of void* objects
            return NULL;

        IDirect3D9* pD3D = Direct3DCreate9(D3D_SDK_VERSION);

        if (!pD3D) // make sure this initialized
            return false;

        IDirect3DDevice9* pDummyDevice = NULL; // "initialize" memory for this

        // options to create dummy device
        D3DPRESENT_PARAMETERS d3dpp = {};
        d3dpp.Windowed = false;
        d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
        d3dpp.hDeviceWindow = GetProcessWindow(); // gets our current window which this functions is made in windowSetup.cpp

        HRESULT dummyDeviceCreated = pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, d3dpp.hDeviceWindow, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &d3dpp, &pDummyDevice);

        if (dummyDeviceCreated != S_OK)
        {
            // may fail in windowed fullscreen mode, trying again with windowed mode
            d3dpp.Windowed = !d3dpp.Windowed;

            dummyDeviceCreated = pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, d3dpp.hDeviceWindow, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &d3dpp, &pDummyDevice);

            if (dummyDeviceCreated != S_OK)
            {
                pD3D->Release();
                return false;
            }
        }

        // scary cursed C-like code
        // but basically *reinterpret_cast<void***>(pDummyDevice) reinterprets the D3D device as a pointer to an array of pointers
        // which sounds complicated, but this basically tells the program to not interpret it as a D3D device, but instead a
        // table of functions related to it, so its all manual to us, pDummyDevice->Present would do the same thing but find where that function
        // is in the array of pointers, call it with the device itself and finally, it would actually execute this function (Present)

        // but since we casted this into a void** (notice the * after reinterpret_cast), this now looks like a big table of virtual functions that
        // we ourselves can dispatch this to any d3d device in the program's space
        // meaning, if we hook the functions, we can get our d3d device since C++ passes in the device before any arguments
        // so EndScene takes no parameters (to us), but when we reinterpret that data as a void**, we must provide that data, meaning EndScene actually
        // has 1 paramater, that being the device, or truthful to the declaration--"this"

        memcpy(pTable, *reinterpret_cast<void***>(pDummyDevice), Size);
        pDummyDevice->Release();

        return true;
	}
}