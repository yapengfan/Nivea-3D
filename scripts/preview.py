"""Render the NLA clips to a compact 12 fps viewing copy using the model's studio."""
from pathlib import Path
import os
import bpy
from mathutils import Vector

root=Path(__file__).resolve().parents[1]
scene=bpy.context.scene
rig=scene.objects["Dragon Girl | Rig"]
anim=rig.animation_data
track=anim.nla_tracks[0]
parts=[(s.name,s.action,round(s.frame_end-s.frame_start)) for s in track.strips]
anim.nla_tracks.remove(track)
track=anim.nla_tracks.new()
for m in list(scene.timeline_markers):scene.timeline_markers.remove(m)
cursor=1
original=scene.camera
# Front for the greeting, rear three-quarter to reveal wing roots, side three-quarter for the tail.
views=((1,.23,.04),(1,.10,.015),(-1,-.24,.07),(-.80,.72,.10))
for (name,action,length),angle in zip(parts,views):
    strip=track.strips.new(name,cursor,action)
    strip.scale=.5
    strip.extrapolation='NOTHING'
    cam=original.copy()
    cam.data=original.data.copy()
    scene.collection.objects.link(cam)
    center=Vector((.008,-.012,.002))
    cam.location=center+Vector(angle).normalized()*3
    cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type='ORTHO'
    cam.data.ortho_scale=1.12
    scene.timeline_markers.new(name,frame=cursor).camera=cam
    cursor+=length//2
scene.frame_start=1
scene.frame_end=cursor-1
scene.frame_set(1)
scene.render.engine="CYCLES"
scene.cycles.samples=8
scene.cycles.use_denoising=True
scene.render.use_persistent_data=True
try:
    pref=bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type='OPTIX'
    pref.refresh_devices()
    if any(d.type=='OPTIX' for d in pref.devices):
        for d in pref.devices:d.use=d.type=='OPTIX'
        scene.cycles.device='GPU'
except (TypeError,RuntimeError):
    pass
scene.render.use_border=False
scene.render.use_crop_to_border=False
scene.render.resolution_x=448
scene.render.resolution_y=560
scene.render.resolution_percentage=100
scene.render.fps=12
scene.render.fps_base=1
scene.render.image_settings.media_type='VIDEO'
scene.render.image_settings.file_format='FFMPEG'
scene.render.ffmpeg.format='MPEG4'
scene.render.ffmpeg.codec='H264'
scene.render.ffmpeg.constant_rate_factor='HIGH'
dest=root/'motion_preview.mp4'
if os.environ.get("DRAGON_TAIL_MOVIE"):
    scene.frame_start=next(marker.frame for marker in scene.timeline_markers if marker.name=='Tail_Sweep')
    dest=Path(os.environ["DRAGON_TAIL_MOVIE"])
scene.render.filepath=str(dest)
bpy.ops.render.render(animation=True)
print('Finished viewing copy',dest)
