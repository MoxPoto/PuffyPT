CudaTracer = {}
CudaTracer.LAMBERTIAN = 0
CudaTracer.SPECULAR = 1
CudaTracer.REFRACT = 2

CudaTracer.TextureRT = GetRenderTarget("CUDATRACER_TEXTURE_RIP", 256, 256)
CudaTracer.Textures = {}

local function sayOutput(...)
    chat.AddText(Color(0, 0, 0), "[", Color(0, 255, 0), "Puffy PT", Color(0, 0, 0), "]", Color(255, 255, 255), ": ", ...)
end

function CudaTracer:AddTexture(textureName)
    if self.Textures[textureName] == true then
        return
    end

    local foundMat = Material(textureName)
    local rawImageData = {}

    print("Textures width: ", foundMat:Width())
    print("Textures height: ", foundMat:Height())

    local originalFlag = foundMat:GetInt("$flags")

    foundMat:SetInt( "$flags", bit.bor( foundMat:GetInt( "$flags" ), 32768 ) )

    render.PushRenderTarget(self.TextureRT)
        cam.Start2D()
            surface.SetDrawColor(255, 255, 255, 255)
            surface.SetMaterial(foundMat)
            surface.DrawTexturedRect(0, 0, 256, 256)
        cam.End2D()

        render.CapturePixels()

        for y = 0, 256 do 
            for x = 0, 256 do 
                local r, g, b = render.ReadPixel(x, y)

                table.insert(rawImageData, r / 255) 
                table.insert(rawImageData, g / 255)  
                table.insert(rawImageData, b / 255) 
            end
        end
    render.PopRenderTarget()

    foundMat:SetInt("$flags", originalFlag)
    tracerSync.UploadTexture(textureName, rawImageData)

    self.Textures[textureName] = true
end

function CudaTracer:AddObject(ent)
    local brdf = self.LAMBERTIAN
    local position = ent:GetPos()
    local color = Vector(ent:GetColor().r / 255, ent:GetColor().g / 255, ent:GetColor().b / 255)
    local emission = ((ent:GetMaterial() == "lights/white") and 95 or 1)

    if ent:GetMaterial() == "debug/env_cubemap_model" then
        brdf = self.SPECULAR
    elseif ent:GetMaterial() == "models/dog/eyeglass" then
        brdf = self.REFRACT
    end

    if brdf == self.REFRACT then 
        color = Vector(1, 1, 1) - color 
    end 

    if ent:GetModel():find("sphere") then 
        local size = ent:GetModelRadius() -- take radius 

    

        sayOutput("Uploading sphere with radius ", tostring(size), "!")

        local id = tracerSync.UploadSphere(ent:GetMaterials()[1], brdf, emission, color, position, size)

        ent.PTRoughness = 0 
        ent.PTIOR = 1.5
        ent.PTID = id
        return 
    end 

    local verts = {}
    -- tfw when
    local objectID = ent:EntIndex()
    local min, max = ent:WorldSpaceAABB()

    local model = util.GetModelMeshes(ent:GetModel())[1]
    local matrix = ent:GetWorldTransformMatrix()

    for k , v in ipairs(model.triangles) do
        local newPos = ent:LocalToWorld(v.pos)
        verts[#verts+1] = newPos
        verts[#verts+1] = v.u 
        verts[#verts+1] = v.v -- wtf
    end

    for k , v in pairs(verts) do
        print(tostring(v))
    end

    local id = tracerSync.UploadMesh(ent:GetMaterials()[1], position, brdf, min, max, color, emission, objectID, verts)
    sayOutput("Uploading mesh with ", tostring(#verts / 3), " triangles!")

    ent.PTRoughness = 0 
    ent.PTIOR = 1.5
    ent.PTID = id
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
                self:AddTexture(v:GetMaterials()[1])
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
    tracerSync.SetCameraAngles(Vector(LocalPlayer():EyeAngles().p, LocalPlayer():EyeAngles().y, 0))
end)

local PREFIX = "/"

hook.Add("OnPlayerChat", "chat_gizmos", function(ply, text)
    if tracerSync == nil then return end 

    if ply == LocalPlayer() then 
        if text:sub(1, 1) == PREFIX then 
            print(text)
            local cmdArgs = string.Split(text, " ")
            local ent = LocalPlayer():GetEyeTrace().Entity 

            if ent.PTIOR == nil or ent.PTRoughness == nil then return end 
            PrintTable(cmdArgs)

            if cmdArgs[1] == "/setRoughness" then 
                local roughness = tonumber(cmdArgs[2])
                ent.PTRoughness = roughness 

                tracerSync.SetObjectLighting(ent.PTID, ent.PTRoughness, ent.PTIOR)
                sayOutput("Set roughness of object ", tostring(ent.PTID), " to ", tostring(roughness))

            elseif cmdArgs[1] == "/setIOR" then 
                local ior = tonumber(cmdArgs[2])
                ent.PTIOR = ior 

                tracerSync.SetObjectLighting(ent.PTID, ent.PTRoughness, ent.PTIOR)
                sayOutput("Set IOR of object ", tostring(ent.PTID), " to ", tostring(ior))
            end
        end 
    end
end)