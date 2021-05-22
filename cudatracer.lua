CudaTracer = {}


function CudaTracer:AddObject(ent)
    local verts = {}
    -- tfw when no color metatable
    local color = Vector(ent:GetColor().r / 255, ent:GetColor().g / 255, ent:GetColor().b / 255)
    local emission = ((ent:GetMaterial() == "lights/white") and 65 or 1)
    local objectID = ent:EntIndex()
    local min, max = ent:WorldSpaceAABB()

    local model = util.GetModelMeshes(ent:GetModel())[1]
    local matrix = ent:GetWorldTransformMatrix()

    for k , v in ipairs(model.triangles) do
        local newPos = ent:LocalToWorld(v.pos)
        verts[#verts+1] = newPos
    end

    for k , v in pairs(verts) do
        print(tostring(v))
    end

    print("Tringle table len: " .. #verts)

    tracerSync.UploadMesh(min, max, color, emission, objectID, verts)
end

function CudaTracer:Remove()
    hook.Remove("Think", "cam_updater")
end

function CudaTracer:AddObjectAt()
    self:AddObject(LocalPlayer():GetEyeTrace().Entity)
end

function CudaTracer:AddMyObjects()
    for k , v in pairs(ents.FindByClass("prop_physics")) do
        if (v.CPPIGetOwner) then
            if (v:CPPIGetOwner() == LocalPlayer()) then
                print("Adding " .. tostring(v))
                self:AddObject(v)
            end
        end
    end
end

function CudaTracer:DoTest()
    require("cudatracer")

    timer.Simple(2, function()
        CudaTracer:AddMyObjects()
    end)
end


hook.Add("Think", "cam_updater", function()
    if tracerSync == nil then return end
    tracerSync.SetCameraPos(LocalPlayer():EyePos())
    tracerSync.SetCameraAngles(LocalPlayer():EyeAngles())
end)