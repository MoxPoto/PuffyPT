#include <GarrysMod/Lua/Interface.h>
#include <vector>
#include <iostream>

#include "syncMain.cuh"
#include "../cpugpu/objects.cuh"
#include <string>

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
	LUA->CheckType(-1, Lua::Type::Vector);

	Vector ourVec = LUA->GetVector(-1);
	CPU::SetCameraAngles(vec3(ourVec.x, ourVec.y, ourVec.z));

	return 0;
}

LUA_FUNCTION(SYNC_UploadMesh) {
	using namespace Tracer;
	LUA->CheckType(-8, Lua::Type::Vector); // Position
	LUA->CheckType(-7, Lua::Type::Number); // BRDF
	LUA->CheckType(-6, Lua::Type::Vector); // AABB Min
	LUA->CheckType(-5, Lua::Type::Vector); // AABB Max
	LUA->CheckType(-4, Lua::Type::Vector); // Color
	LUA->CheckType(-3, Lua::Type::Number); // Emission
	LUA->CheckType(-2, Lua::Type::Number); // Game Id
	LUA->CheckType(-1, Lua::Type::Table); // Vertices

	int ourID = CPU::AddTracerObject(CPU::TriangleMesh);

	size_t len = LUA->ObjLen();
	printf("[host] Received table with length: %s\n", std::to_string(len).c_str());
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

	printf("[host] Pushed %d triangles\n", verts.size());

	for (size_t i = 0; i < verts.size(); i += 3) {
		vec3 v1(verts[i].x, verts[i].y, verts[i].z);
		vec3 v2(verts[i + 1].x, verts[i + 1].y, verts[i + 1].z);
		vec3 v3(verts[i + 2].x, verts[i + 2].y, verts[i + 2].z);

		// printf("[host] v1: %.2f, %.2f, %.2f\n", v1.x(), v1.y(), v1.z());
		// printf("[host] v1 - verts : %.2f, %.2f, %.2f\n", verts[i].x, verts[i].y, verts[i].z);
		
		CPU::CommandError err = CPU::InsertObjectTri(ourID, v1, v2, v3);
		if (err != CPU::CommandError::Success) {
			std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
		}
		
	}

	LUA->Pop(2);

	Vector pos = LUA->GetVector(-7);
	int brdfType = static_cast<int>(LUA->GetNumber(-6));
	Vector nMin = LUA->GetVector(-5);
	Vector nMax = LUA->GetVector(-4);
	Vector color = LUA->GetVector(-3);
	float emission = static_cast<float>(LUA->GetNumber(-2));
	int gameID = static_cast<int>(LUA->GetNumber(-1));

	CPU::CommandError cmdErr = CPU::SetColorEmission(ourID, vec3(color.x, color.y, color.z), emission);

	if (cmdErr != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr2 = CPU::ComputeMeshAccel(ourID, vec3(nMin.x, nMin.y, nMin.z), vec3(nMax.x, nMax.y, nMax.z));

	if (cmdErr2 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr3 = CPU::SetBRDF(ourID, static_cast<BRDF>(brdfType));

	if (cmdErr3 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr4 = CPU::SetObjectPosition(ourID, vec3(pos.x, pos.y, pos.z));

	if (cmdErr4 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}


	LUA->PushNumber(ourID);

	return 1;
}

LUA_FUNCTION(SYNC_UploadSphere) {
	using namespace Tracer;

	LUA->CheckType(-5, Lua::Type::Number); // BRDF
	LUA->CheckType(-4, Lua::Type::Number); // Emission
	LUA->CheckType(-3, Lua::Type::Vector); // Color
	LUA->CheckType(-2, Lua::Type::Vector); // Center
	LUA->CheckType(-1, Lua::Type::Number); // Size

	int ourID = CPU::AddTracerObject(CPU::Sphere);

	int brdfType = static_cast<int>(LUA->GetNumber(-5));
	float emission = static_cast<float>(LUA->GetNumber(-4));
	Vector color = LUA->GetVector(-3);
	Vector pos = LUA->GetVector(-2);
	float size = static_cast<float>(LUA->GetNumber(-1));

	CPU::CommandError cmdErr = CPU::SetColorEmission(ourID, vec3(color.x, color.y, color.z), emission);

	if (cmdErr != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr1 = CPU::SetBRDF(ourID, static_cast<BRDF>(brdfType));

	if (cmdErr1 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr2 = CPU::SetObjectPosition(ourID, vec3(pos.x, pos.y, pos.z));

	if (cmdErr2 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	CPU::CommandError cmdErr3 = CPU::SetSphereSize(ourID, size);

	if (cmdErr3 != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
	}

	LUA->PushNumber(ourID);

	return 1;
}

LUA_FUNCTION(SYNC_SetLighting) {
	using namespace Tracer;

	LUA->CheckType(-3, Lua::Type::Number); // ID
	LUA->CheckType(-2, Lua::Type::Number); // Roughness
	LUA->CheckType(-1, Lua::Type::Number); // IoR

	int id = static_cast<int>(LUA->GetNumber(-3));
	float roughness = static_cast<float>(LUA->GetNumber(-2));
	float ior = static_cast<float>(LUA->GetNumber(-1));

	LightingOptions newOptions;
	newOptions.roughness = roughness;
	newOptions.ior = ior;

	CPU::CommandError cmdErr = CPU::CommitObjectLighting(id, newOptions);

	if (cmdErr != CPU::CommandError::Success) {
		std::cout << "Command error hit on line " << __LINE__ << "!!!\n";
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
			TABLE_FUNC("UploadSphere", SYNC_UploadSphere);
			TABLE_FUNC("UploadMesh", SYNC_UploadMesh);
			TABLE_FUNC("SetObjectLighting", SYNC_SetLighting);

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