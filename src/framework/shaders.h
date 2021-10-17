#ifndef SHADER_H
#define SHADER_H

#include <slang.h>
#include <slang-com-ptr.h>

#include <wrl/client.h>
#include <d3d11.h>


class Shaders {
private:
	Slang::ComPtr<slang::IGlobalSession> globalSession;
	Slang::ComPtr<slang::ISession> mainSession;
public:
	Microsoft::WRL::ComPtr<ID3DBlob> Compile(const char* module, const char* entryPoint);

	Shaders();
	~Shaders();
};

#endif