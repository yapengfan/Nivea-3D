"""Expand the existing 56-bone character rig without changing its bind pose."""
import bpy
import sys,json
import numpy as np
from pathlib import Path
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))


def evaluated_points(scene):
    deps=bpy.context.evaluated_depsgraph_get()
    result={}
    for o in scene.objects:
        if o.type!="MESH":continue
        evo=o.evaluated_get(deps)
        vertices=evo.data.vertices
        xyz=np.empty((len(vertices),3),'f')
        vertices.foreach_get("co",xyz.ravel())
        m=np.asarray(evo.matrix_world)
        result[o.name]=xyz@m[:3,:3].T+m[:3,3]
    return result


def edit_skeleton(rig,parts,modules):
    arm=rig.data
    bpy.context.view_layer.objects.active=rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for key,items in parts.items():
        hints=modules[key].roll_hints(bpy.context.scene) if hasattr(modules[key],"roll_hints") else {}
        group=arm.collections.new({"cloth":"Garments | skirts and drapes","hair":"Hair | face framing and back",
                                   "tail":"Tail | continuous FK","wrist":"Cuffs | hanging tassels"}[key])
        for name,(head,tail,parent) in items.items():
            if name in arm.edit_bones:raise ValueError("Existing bone: "+name)
            new=arm.edit_bones.new(name)
            new.head=head
            new.tail=tail
            new.parent=arm.edit_bones[parent]
            new.use_connect=(new.head-new.parent.tail).length < .00001
            tangent=(new.tail-new.head).normalized()
            direction=Vector((0,0,1)) if key=="tail" else Vector((1,0,0))
            if abs(tangent.dot(direction))>.98:
                direction=Vector((0,1,0))
            new.align_roll(Vector(hints.get(name,direction)))
            new.head_radius=new.tail_radius=max(.0006,new.length*.07)
            for old in list(new.collections):old.unassign(new)
            group.assign(new)
            if key=="tail":new.color.palette="THEME04"
            elif key=="hair":new.color.palette="THEME13"
            else:new.color.palette="THEME09"
    bpy.ops.object.mode_set(mode='OBJECT')
    for partspec in parts.values():
        for name in partspec:
            rig.pose.bones[name].rotation_mode='XYZ'


def main():
    import tail,hair,cloth,wrist,motions
    scene=bpy.context.scene
    if Path(bpy.data.filepath).name!="character_rig_v1.blend":
        raise ValueError("Open the refined character's basic rig first.")
    rig=scene.objects["Dragon Girl | Rig"]
    if len(rig.data.bones)!=56:raise ValueError("Unexpected input skeleton")
    if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    before=evaluated_points(scene)
    mods={'tail':tail,'hair':hair,'cloth':cloth,'wrist':wrist}
    parts={key:mod.bones(scene) for key,mod in mods.items()}
    edit_skeleton(rig,parts,mods)
    for key,module in mods.items():
        print("Bind",key,len(parts[key]),module.apply(scene,rig,parts[key]),flush=True)
    bpy.context.view_layer.update()
    after=evaluated_points(scene)
    error=max(np.linalg.norm(before[o]-after[o],axis=1).max() for o in before)
    if error>.0001:raise ValueError(f"Rest surface moved: {error}")
    actions=motions.build(scene,rig,parts)
    bpy.context.view_layer.update()
    for o in bpy.context.selected_objects:o.select_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='DESELECT')
    # Space immediately previews the finished character; overlays can be toggled to edit the rig.
    for area in bpy.context.screen.areas:
        if area.type=="VIEW_3D":
            area.spaces.active.overlay.show_overlays=False
        if area.type=="DOPESHEET_EDITOR":
            for region in area.regions:
                if region.type=="WINDOW":
                    with bpy.context.temp_override(area=area,region=region):
                        try:bpy.ops.action.view_all()
                        except RuntimeError:pass
    scene.sync_mode="FRAME_DROP"
    rig["动作播放"]="播放时间线预览四段动作。按骨骼集合编辑，开启视图叠加层显示骨架。"
    scene.render.filepath="//renders/"
    bpy.context.preferences.filepaths.save_version=0
    dest=ROOT/"character_rig_v2_actions.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(dest),compress=True,relative_remap=True)
    print("SAVED",dest, "Bones",len(rig.data.bones),"Actions",actions,"Last frame",scene.frame_end,"Rest delta",error)


if __name__=='__main__':main()
