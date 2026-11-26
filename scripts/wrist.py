"""FK for the two hanging beads and cords, pivoting at the authored cuff eyelets."""
import numpy as np


def fittings(scene):
    for side,clasp,bead,tassel in (("L",113,112,111),("R",34,33,31)):
        obj=scene.objects["Wrist | cuff and tassel."+side]
        matrix=np.asarray(obj.matrix_world)
        p=np.array([v.co[:] for v in obj.data.vertices]) @ matrix[:3,:3].T + matrix[:3,3]
        islands=np.array([v.value for v in obj.data.attributes['Source_Island'].data])
        eyelet=np.flatnonzero(islands==clasp)
        cord=np.flatnonzero(islands==tassel)
        moving=np.flatnonzero((islands==tassel)|(islands==bead))
        pivot=p[eyelet[np.argmin(p[eyelet,2])]].copy()
        end=p[cord[np.argmin(p[cord,2])]].copy()
        yield side,obj,p,eyelet,moving,pivot,end


def bones(scene):
    return {"WristDrop.01."+s:(pivot,end,"Forearm."+s)
            for s,obj,p,eyelet,moving,pivot,end in fittings(scene)}


def apply(scene,rig,defs):
    counts={}
    for side,obj,p,eyelet,moving,pivot,end in fittings(scene):
        name="WristDrop.01."+side
        bone_names=rig.data.bones.keys()
        existing=[g for g in obj.vertex_groups if g.name in bone_names]
        hanger=obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        ids=moving.tolist()
        for g in existing:g.remove(ids)
        hanger.add(ids,1.,'REPLACE')
        # Match the skin exactly at the open diamond's lower tip. The upper
        # half of the diamond stays on the cuff with its original wrist blend.
        anchor=obj.vertex_groups["Forearm."+side]
        for vert in eyelet:
            t=float(np.clip((p[vert,2]-pivot[2])/.013,0.,1.))
            u=t*t*(3.-2.*t)
            if u>=1:continue
            old={g.group:g.weight for g in obj.data.vertices[int(vert)].groups}
            for g in existing:
                w=old.get(g.index,0.)*u+(1.-u if g==anchor else 0.)
                if w>1.e-7:g.add([int(vert)],w,'REPLACE')
                elif g.index in old:g.remove([int(vert)])
        counts[name]=len(ids)
    return {"weighted_beads_and_cords":counts,"pivots":"lower points of the original diamond eyelets"}
