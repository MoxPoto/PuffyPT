#ifndef SHADER_H
#define SHADER_H

#include <slang.h>
#include <slang-com-ptr.h>

class Shaders {
private:
	Slang::ComPtr<slang::IGlobalSession> globalSession;
	Slang::ComPtr<slang::ISession> mainSession;
public:
	Slang::ComPtr<slang::IBlob> Compile(const char* module, const char* entryPoint);

	Shaders();
	~Shaders();
};

#endif