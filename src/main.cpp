#include <GarrysMod/Lua/Interface.h>
#include <framework/framework.h>
#include <globals.h>

std::shared_ptr<Framework> framework;

GMOD_MODULE_OPEN() {
	framework = std::make_shared<Framework>();
	return 0;
}

GMOD_MODULE_CLOSE() {
	// TODO: actually make the program..
	framework.reset(); // Kill the framework
	return 0;
}