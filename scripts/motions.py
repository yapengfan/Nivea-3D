"""Four FK clips. World-axis rotations are converted to the individual bones' rest axes."""
import math
import bpy
from mathutils import Euler, Quaternion, Vector

CYCLE = 2*math.pi
CLIPS = (
    ("Idle_Breeze","待机 · 微风",96),
    ("Greeting","抬手招呼",80),
    ("Wing_Display","展翼轻拍",72),
    ("Tail_Sweep","回首摆尾",88),
)


def fade(a,b,t):
    x=max(0.,min(1.,(t-a)/(b-a)))
    return x*x*(3-2*x)


def rot(x=0,y=0,z=0):
    return Euler(tuple(math.radians(a) for a in (x,y,z))).to_quaternion()


def world_key(bone,rotation,frame,previous):
    basis=bone.bone.matrix_local.to_quaternion()
    local=basis.inverted() @ rotation @ basis
    angles=local.to_euler("XYZ",previous.get(bone.name,Euler((0,0,0))))
    previous[bone.name]=angles.copy()
    bone.rotation_euler=angles
    bone.keyframe_insert(data_path="rotation_euler",frame=frame,group=bone.name)


def hierarchy(defs,rig):
    result=[]
    for kind,part in defs.items():
        for name in part:
            n=name
            depth=0
            while part[n][2] in part:
                n=part[n][2]
                depth+=1
            chain=rig.data.bones[n]
            p=chain.head_local
            az=math.atan2(p.y+.012,p.x-.125)
            result.append((kind,name,n,depth,az))
    return result


def global_rotation(bone, poses):
    rotation=Quaternion()
    for part in [*reversed(bone.parent_recursive),bone]:
        rotation=rotation @ poses.get(part.name,Quaternion())
    return rotation


def motion(kind,t,rig,secondary):
    phase=CYCLE*t
    # All clips meet at the authored standing pose; the pauses give the gesture time to read.
    env=math.sin(math.pi*t)**2
    lift=fade(.04,.36,t)*(1-fade(.68,.98,t)) if kind=="Greeting" else 0.
    openwing=fade(.05,.30,t)*(1-fade(.75,.98,t)) if kind=="Wing_Display" else 0.
    sweep=env if kind=="Tail_Sweep" else 0.
    nod=math.sin(phase)*env
    wave=math.sin(2*phase+.2)*lift
    beats=math.sin(2*phase-1.0)*openwing
    amount={"Idle_Breeze":1.,"Greeting":1.45,"Wing_Display":1.5,"Tail_Sweep":2.1}[kind]
    turn=7*lift+10*sweep
    poses={
        "Spine":rot(x=.8*nod,y=-.9*nod,z=.25*turn),
        "Chest":rot(x=-1.4*lift,y=-1.6*env-1.6*openwing,z=.4*turn),
        "Neck":rot(x=1.5*lift,y=2*env,z=.3*turn),
        "Head":rot(x=-3.5*lift,y=1.5*nod+3*lift,z=2*nod+turn),
    }
    for side,sign in (("L",1),("R",-1)):
        poses["Shoulder."+side]=rot(x=sign*(3.8*lift*(side=="L")+1.3*openwing),z=-.8*sign*env)
        poses["WingRoot."+side]=rot(x=sign*(.9*nod+18*openwing+9*beats),z=-sign*9*openwing)
        poses["WingMid."+side]=rot(x=sign*(.7*nod+7*openwing+6*math.sin(2*phase-1.45)*openwing),
                                   z=-sign*5*openwing)
        poses["WingTip."+side]=rot(x=sign*(.6*nod+4*openwing+6*math.sin(2*phase-1.9)*openwing))
    if kind=="Greeting":
        up=rig.data.bones["UpperArm.L"]
        fore=rig.data.bones["Forearm.L"]
        u=(up.tail_local-up.head_local).normalized()
        f=(fore.tail_local-fore.head_local).normalized()
        # Match the actual arm lengths while turning the palm toward the viewer.
        qu=Quaternion().slerp(u.rotation_difference(Vector((.18,.76,-.62)).normalized()),lift)
        poses["UpperArm.L"]=qu
        fore_goal=Vector((.38,.13+.12*wave,.91)).normalized()
        qb=Quaternion().slerp(f.rotation_difference(fore_goal),lift)
        poses["Forearm.L"]=qu.inverted() @ qb
        poses["Hand.L"]=rot(x=-10*wave,y=5*lift,z=9*wave)
        poses["UpperArm.R"]=rot(x=-3*lift,y=-3*lift)
        for j,nm in enumerate(("Index","Middle","Ring","Pinky")):
            for k in (1,2,3):
                pb=rig.pose.bones[f"{nm}.{k:02d}.L"]
                basis=pb.bone.matrix_local.to_quaternion()
                poses[pb.name]=basis @ rot(x=(3+j*2)*lift+1.2*wave) @ basis.inverted()
    for part,name,root,d,az in secondary:
        side=1 if rig.data.bones[root].head_local.y>-.012 else -1
        lag=.39*d
        delayed=env*math.sin(phase-.45-lag+.22*math.sin(az))
        if part=="tail":
            # The curled tail trails the pelvis. Bias its large sweep toward the space behind the dress.
            rear=sweep*(3.3+3.8*math.sin(phase*.98-.44*d))
            poses[name]=rot(x=amount*.8*delayed+sweep*2.7*math.sin(phase-.52*d),
                            y=amount*.46*delayed+sweep*2.8*math.sin(phase-.50*d+.5),
                            z=amount*.92*delayed+rear)
            if kind=="Tail_Sweep":
                # Ease the returning curl past the inside of the long left sleeve.
                returning=fade(.65,.75,t)*(1.-fade(.82,.92,t))
                poses[name]=Quaternion().slerp(poses[name],1.-.12*returning)
        elif part=="hair":
            short="bang" in root.lower() or "fringe" in root.lower()
            gain=.36 if short else 1.
            poses[name]=rot(x=gain*(amount*1.5*delayed-.8*side*lift),
                            y=gain*(.8*delayed+1.05*beats),
                            z=gain*(-.17*turn*env+amount*.9*delayed))
        else:
            sleeve=("sleeve" in root.lower() or "cascad" in root.lower())
            tassel=any(word in root.lower() for word in ("tassel","ribbon","lariat","drop"))
            follow=0
            if sleeve and side==1:
                follow=.8*wave
            gain=1.55 if tassel else 1
            poses[name]=rot(x=gain*(amount*.9*delayed+follow),
                            y=gain*(amount*.52*delayed+.85*math.cos(az)*beats),
                            z=gain*(amount*.40*delayed-.4*nod))
            if sleeve and side==1 and kind=="Greeting":
                # Let the sleeve mouth follow the arm, then progressively return the long panel to gravity.
                correction=Quaternion().slerp(qu.inverted(),(.37,.38,.19,.05)[d])
                poses[name]=correction @ poses[name]
            if part=="wrist" and not d:
                # Keep the moving pivot on its clasp while the hanging cord swings below it.
                parent=rig.data.bones[name].parent
                world=global_rotation(parent,poses)
                cancel=Quaternion().slerp(world.inverted(),.98)
                poses[name]=cancel @ rot(x=.85*wave+1.8*delayed,y=1.8*wave)
    return poses


def build(scene,rig,defs):
    if rig.animation_data and (rig.animation_data.action or rig.animation_data.nla_tracks):
        raise RuntimeError("The input rig already has animation")
    anim=rig.animation_data_create()
    secondary=hierarchy(defs,rig)
    actions=[]
    for kind,title,length in CLIPS:
        action=bpy.data.actions.new(kind)
        action.use_fake_user=True
        action["动作"]=title
        anim.action=action
        prev={}
        for frame in range(1,length+2,4):
            for name,q in motion(kind,(frame-1)/length,rig,secondary).items():
                world_key(rig.pose.bones[name],q,frame,prev)
        action.use_frame_range=True
        action.frame_start=1
        action.frame_end=length+1
        action.use_cyclic=(kind in ("Idle_Breeze","Tail_Sweep"))
        anim.action=None
        actions.append(action)
    for pb in rig.pose.bones:
        pb.matrix_basis.identity()
        pb.rotation_euler=(0,0,0)
    track=anim.nla_tracks.new()
    track.name="动作演示 | Gesture reel"
    cursor=1
    for action,(name,title,length) in zip(actions,CLIPS):
        strip=track.strips.new(name,cursor,action)
        strip.name=name
        strip.extrapolation='NOTHING'
        strip.blend_type='REPLACE'
        scene.timeline_markers.new(title,frame=cursor)
        cursor+=length
    scene.render.fps=24
    scene.render.fps_base=1
    scene.frame_start=1
    scene.frame_end=cursor-1
    scene.frame_set(1)
    return [a.name for a in actions]
