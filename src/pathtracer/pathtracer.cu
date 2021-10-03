#include <pathtracer/pathtracer.cuh>
#include <cuda_runtime.h>
#include <imgui.h>

__host__ void Pathtracer::ImGuiUpdate() {
	ImGui::Text("Hello from the pathtrar clas!! ! !! !!");
	ImGui::Button("Funky button");
}

__host__ Pathtracer::Pathtracer() {

}