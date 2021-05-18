#include <GarrysMod/Lua/Interface.h>
#include <vector>
#include <iostream>

#include "syncMain.cuh"
#include "../cpugpu/objects.cuh"

#define SYNC_NAME "tracerSync"
#define TABLE_FUNC(name, cfuncName) LUA->PushString(name); LUA->PushCFunction(cfuncName); LUA->SetTable(-3);

using namespace GarrysMod;

LUA_FUNCTION(SYNC_SetCameraPos) {
	using namespace Tracer;
	LUA->CheckType(-1, Lua::Type::Vector);

	Vector ourVec = LUA->GetVector(-1);
	CPU::SetCameraPos(ourVec.x, ourVec.y, ourVec.z);

	return 0;
}

LUA_FUNCTION(SYNC_SetCameraAngles) {
	using namespace Tracer;
	LUA->CheckType(-1, Lua::Type::Angle);

	QAngle ourVec = LUA->GetAngle(-1);
	CPU::SetCameraAngles(ourVec.x, ourVec.y, ourVec.z);

	return 0;
}

LUA_FUNCTION(SYNC_UploadMesh) {
	using namespace Tracer;
	LUA->CheckType(-4, Lua::Type::Vector); // Color
	LUA->CheckType(-3, Lua::Type::Number); // Emission
	LUA->CheckType(-2, Lua::Type::Number); // Game Id
	LUA->CheckType(-1, Lua::Type::Table); // Vertices

	Vector color = LUA->GetVector(-4);
	float emission = static_cast<float>(LUA->GetNumber(-3));
	int gameID = static_cast<int>(LUA->GetNumber(-2));
	int ourID = CPU::AddTracerObject(CPU::TriangleMesh);

	std::cout << "Got Tracer ID: " << ourID << "\n";

	Sync::Prop newProp;
	newProp.gameID = gameID;
	newProp.tracerID = ourID;

	Sync::props.push_back(newProp);

	LUA->Pop(3);
	
	size_t len = LUA->ObjLen();

	std::vector<Vector> verts;

	for (int index = 0; index <= len; index++) {
		// Our actual index will be +1 because Lua 1 indexes tables.
		int actualIndex = index + 1;
		// Push our target index to the stack.
		LUA->PushNumber(actualIndex);
		// Get the table data at this index (and not get the table, which is what I thought this did.)
		LUA->GetTable(-2);
		// Check for the sentinel nil element.
		if (LUA->GetType(-1) == GarrysMod::Lua::Type::Nil) break;
		// Get it's value.
		verts.push_back(LUA->GetVector());
		// Pop it off again.
		LUA->Pop(1);
	}

	for (int i = 0; i < verts.size(); i += 3) {
		vec3 v1(verts[i].x, verts[i].y, verts[i].z);
		vec3 v2(verts[i + 1].x, verts[i + 1].y, verts[i + 1].z);
		vec3 v3(verts[i + 2].x, verts[i + 2].y, verts[i + 2].z);

		CPU::CommandError err = CPU::InsertObjectTri(ourID, v1, v2, v3);
		if (err != CPU::CommandError::Success) {
			std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
		}
	}

	return 0;
}

namespace Tracer {
	namespace Sync {
		std::vector<Prop> props;

		void Initialize(Lua::ILuaBase* LUA) {
			LUA->PushSpecial(Lua::SPECIAL_GLOB);
			LUA->PushString(SYNC_NAME);

			LUA->CreateTable();

			TABLE_FUNC("SetCameraPos", SYNC_SetCameraPos);
			TABLE_FUNC("SetCameraAngles", SYNC_SetCameraAngles);
			TABLE_FUNC("UploadMesh", SYNC_UploadMesh);

			LUA->SetTable(-3);

			LUA->Pop(1); // Pop glob off
		}
		
		void Deinitialize(Lua::ILuaBase* LUA) {
			LUA->PushSpecial(Lua::SPECIAL_GLOB);
			LUA->PushNil();
			LUA->SetField(-2, SYNC_NAME);
			
			LUA->Pop(1); // Pop glob off
		}
	}
}