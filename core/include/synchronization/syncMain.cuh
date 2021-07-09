#ifndef SYNC_MAIN_H
#define SYNC_MAIN_H
#include <GarrysMod/Lua/Interface.h>
#include <vector>

using namespace GarrysMod;

namespace Sync {
	struct Prop {
		int tracerID;
		int gameID;
	};
		
	extern std::vector<Prop> props;

	extern void Initialize(Lua::ILuaBase* LUA);
	extern void Deinitialize(Lua::ILuaBase* LUA);
}

#endif