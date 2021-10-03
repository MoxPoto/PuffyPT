#include <GarrysMod/Lua/Interface.h>
#include <framework/framework.h>
#include <globals.h>

bool alive = true;
static std::shared_ptr<Framework> framework;

GMOD_MODULE_OPEN() {
	framework = std::make_shared<Framework>();
	return 0;
}

GMOD_MODULE_CLOSE() {
	printf("Closing Puffy PT..\n");
	framework.reset();
	return 0;
}