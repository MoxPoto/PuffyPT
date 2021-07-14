local FILE_PREFIX = "C:\\pathtracer\\pbr\\materials\\"
local FILE_SUFFIX = "_mrao.png"

CudaTracer = {}
CudaTracer.LAMBERTIAN = 0
CudaTracer.SPECULAR = 1
CudaTracer.REFRACT = 2

CudaTracer.TextureRT = GetRenderTarget("CUDATRACER_TEXTURE_RIP", 256, 256)
CudaTracer.TextureMat = CreateMaterial("CudaTracerMaterialSoItDoesntGetAffectedByDamnedBakedShadows", "UnlitGeneric", {
    ["$basetexture"] = ""
})

CudaTracer.TextureMat:SetInt( "$flags", bit.bor( CudaTracer.TextureMat:GetInt( "$flags" ), 32768 ) )

CudaTracer.Textures = {}

local function sayOutput(...)
    chat.AddText(Color(0, 0, 0), "[", Color(0, 255, 0), "Puffy PT", Color(0, 0, 0), "]", Color(255, 255, 255), ": ", ...)
end

local customMaterial = CreateMaterial( "exampsle_drt_mdat", "UnlitGeneric", {
	["$basetexture"] = CudaTracer.TextureRT:GetName(), -- You can use "example_rt" as well
} )

hook.Add( "HUDPaint", "ExampleDraw", function()
	surface.SetDrawColor( 255, 255, 255, 255 )
	surface.SetMaterial( customMaterial )
	surface.DrawTexturedRect( 0, 0, customMaterial:GetTexture( "$basetexture" ):Width(), customMaterial:GetTexture( "$basetexture" ):Height() )
end )

function CudaTracer:AddTexture(textureName, isBase)
    isBase = isBase or false 

    if self.Textures[textureName] == true then
        return
    end

    if isBase then 
        self.TextureMat:SetTexture("$basetexture", textureName)
    else
        local foundMat = Material(textureName)
        self.TextureMat:SetTexture("$basetexture", foundMat:GetTexture("$basetexture"))
    end

    local rawImageData = {}

    render.PushRenderTarget(self.TextureRT)
        cam.Start2D()
            surface.SetDrawColor(255, 255, 255, 255)
            surface.SetMaterial(self.TextureMat)
            surface.DrawTexturedRect(0, 0, 256, 256)
        render.CapturePixels()

        for y = 0, 255 do 
            for x = 0, 255 do 
                local r, g, b = render.ReadPixel(x, y)

                table.insert(rawImageData, r / 255) 
                table.insert(rawImageData, g / 255)  
                table.insert(rawImageData, b / 255) 

                surface.SetDrawColor(r, g, b, 255)
                draw.NoTexture()
                surface.DrawRect(x, y, 1, 1)
    
            end
        end

        cam.End2D()
    render.PopRenderTarget()



    tracerSync.UploadTexture(textureName, rawImageData)

    self.Textures[textureName] = true
end

function CudaTracer:AddObject(ent)
    local brdf = self.LAMBERTIAN
    local position = ent:GetPos()
    local color = Vector(ent:GetColor().r / 255, ent:GetColor().g / 255, ent:GetColor().b / 255)
    local emission = ((ent:GetMaterial() == "lights/white") and 95 or 1)
    local material = ""

    if ent:GetMaterial() == "" then
        material = ent:GetMaterials()[1]
    else
        material = ent:GetMaterial()
    end

    if ent:GetModel():find("sphere") then 
        local size = ent:GetModelRadius() -- take radius 

        sayOutput("Uploading sphere with radius ", tostring(size), "!")

        local id = tracerSync.UploadSphere(material, brdf, emission, color, position, size)

        ent.PTID = id
        ent.PTLighting = {
            Metalness = 0,
            BRDF = 0,
            IOR = 1.5,
            Transmission = 0,
            Roughness = 0
        }

        return 
    end 

    local verts = {}
    -- tfw when
    local objectID = ent:EntIndex()
    local min, max = ent:WorldSpaceAABB()

    local model = util.GetModelMeshes(ent:GetModel())[1]
    local normalGlobalization = ent:GetWorldTransformMatrix()
    normalGlobalization:SetTranslation(Vector(0, 0, 0))

    for k , v in ipairs(model.triangles) do
        local newPos = ent:LocalToWorld(v.pos)
        verts[#verts+1] = newPos
        verts[#verts+1] = v.u 
        verts[#verts+1] = v.v -- wtf
        verts[#verts+1] = normalGlobalization * v.normal

        local tangent, binormal = v.tangent, v.binormal 
        if tangent then 
            if binormal == nil then 
                binormal = v.tangent:Cross(v.normal)
            end
        else 
            tangent = Vector(0, 0, 0)
            binormal = Vector(0, 0, 0)
        end

        verts[#verts + 1] = normalGlobalization * tangent
        verts[#verts + 1] = normalGlobalization * binormal
    end

    local id = tracerSync.UploadMesh(material, position, brdf, min, max, color, emission, objectID, verts)
    sayOutput("Uploading mesh with ", tostring(#verts / (4 * 3)), " triangles!")

    ent.PTID = id
    ent.PTLighting = {
        Metalness = 0,
        BRDF = 0,
        IOR = 1.5,
        Transmission = 0,
        Roughness = 0
    }
    
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

                local material = ""

                if v:GetMaterial() == "" then
                    material = v:GetMaterials()[1]
                else
                    material = v:GetMaterial()
                end

                local mraoPath = FILE_PREFIX .. material:gsub("/", "\\") .. FILE_SUFFIX
                local tempMaterial = Material(material)

                local normalMapPath = tempMaterial:GetString("$bumpmap")

                self:AddTexture(material)
                self:AddObject(v)

                if type(normalMapPath) == "string" then 
                    self:AddTexture(normalMapPath, true)
                else 
                    normalMapPath = "_no_normal_map"
                end

                tracerSync.SetPBR(v.PTID, "", normalMapPath)
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

            if ent.PTID == nil then return end 
            PrintTable(cmdArgs)
            
            if cmdArgs[1] == "/set" then
                local indexWanted = tostring(cmdArgs[2])

                if ent.PTLighting[indexWanted] then 
                    local valueWanted = tonumber(cmdArgs[3])

                    print(indexWanted, valueWanted)
                    ent.PTLighting[indexWanted] = valueWanted 

                    sayOutput(string.format("Set %s to %.2f", indexWanted, valueWanted))
                    
                    tracerSync.SetObjectLighting(ent.PTID, ent.PTLighting)
                end
            end
        end 
    end
end)
