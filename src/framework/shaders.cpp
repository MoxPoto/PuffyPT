#include <framework/shaders.h>
#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#include <stdio.h>
#include <wrl/client.h>

using namespace Microsoft;

Shaders::Shaders() {
	// Initialize Slang

	slang::createGlobalSession(globalSession.writeRef());
	slang::SessionDesc sessionDesc;
	slang::TargetDesc targets;
	targets.format = SLANG_HLSL;
	targets.profile = globalSession->findProfile("cs_4_0");

	sessionDesc.targets = &targets;
	sessionDesc.targetCount = 1;

	const char* searchPaths[] = { "C:/puffypt/" };
	sessionDesc.searchPaths = searchPaths;
	sessionDesc.searchPathCount = 1;

	globalSession->createSession(sessionDesc, mainSession.writeRef());
}

WRL::ComPtr<ID3DBlob> Shaders::Compile(const char* module, const char* entryPoint) {
	WRL::ComPtr<ID3DBlob> compiledBytecode;
	WRL::ComPtr<ID3DBlob> compilerErrors;

	Slang::ComPtr<slang::IBlob> diagnosticBlob;

	Slang::ComPtr<slang::IModule> slangModule(mainSession->loadModule(module, diagnosticBlob.writeRef()));
	Slang::ComPtr<slang::IEntryPoint> slangEntry;

	if (slangModule == nullptr || slangEntry == nullptr) {
		if (diagnosticBlob) {
			printf("[%s - %s] Shader diagnostic: %s\n", module, entryPoint, reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer()));
		}

		return compiledBytecode;
	}

	slangModule->findEntryPointByName(entryPoint, slangEntry.writeRef());

	slang::IComponentType* components[] = { slangModule, slangEntry };
	Slang::ComPtr<slang::IComponentType> program;

	mainSession->createCompositeComponentType(components, 2, program.writeRef());

	Slang::ComPtr<slang::IBlob> kernelBlob;
	program->getEntryPointCode(0, 0, kernelBlob.writeRef(), diagnosticBlob.writeRef());

	slangModule->release();
	slangEntry->release();
	program->release();

	if (diagnosticBlob) {
		printf("[%s - %s] Shader diagnostic: %s\n", module, entryPoint, reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer()));
	}
	
	// Now, it may look like we're done, but this is actually returning slang -> hlsl, not slang -> hlsl bytecode obviously

	HRESULT compileCode = D3DCompile(kernelBlob->getBufferPointer(), kernelBlob->getBufferSize(), NULL, NULL, NULL, entryPoint, "cs_4_0", D3DCOMPILE_OPTIMIZATION_LEVEL2, 0, compiledBytecode.GetAddressOf(), compilerErrors.GetAddressOf());

	if (compilerErrors) {
		printf("[%s - %s] !! ERROR FROM D3D11 !!\n%s\n", module, entryPoint, reinterpret_cast<const char*>(compilerErrors->GetBufferPointer()));
	}

	if (compileCode != S_OK) {
		printf("[%s - %s] Couldn't compile slang output!\n", module, entryPoint);
	}

	return compiledBytecode;
}

Shaders::~Shaders() {
	mainSession->release();
	globalSession->release();
}