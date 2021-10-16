#include <framework/shaders.h>
#include <slang.h>
#include <slang-com-ptr.h>
#include <slang-com-helper.h>

#include <d3d11.h>

Shaders::Shaders() {
	// Initialize Slang

	slang::createGlobalSession(globalSession.writeRef());
	slang::SessionDesc sessionDesc;
	slang::TargetDesc targets;
	targets.format = SLANG_HLSL;
	targets.profile = globalSession->findProfile("cs_5_0");

	sessionDesc.targets = &targets;
	sessionDesc.targetCount = 1;

	const char* searchPaths[] = { "src/renderer/" };
	sessionDesc.searchPaths = searchPaths;
	sessionDesc.searchPathCount = 1;

	globalSession->createSession(sessionDesc, mainSession.writeRef());
}

Slang::ComPtr<slang::IBlob> Shaders::Compile(const char* module, const char* entryPoint) {
	Slang::ComPtr<slang::IModule> slangModule(mainSession->loadModule(module));
	Slang::ComPtr<slang::IEntryPoint> slangEntry;

	slangModule->findEntryPointByName(entryPoint, slangEntry.writeRef());
	
	slang::IComponentType* components[] = { slangModule, slangEntry };
	Slang::ComPtr<slang::IComponentType> program;

	mainSession->createCompositeComponentType(components, 2, program.writeRef());

	Slang::ComPtr<slang::IBlob> kernelBlob;
	program->getEntryPointCode(0, 0, kernelBlob.writeRef());

	slangModule->release();
	slangEntry->release();
	program->release();

	return kernelBlob;
}

Shaders::~Shaders() {
	mainSession->release();
	globalSession->release();
}