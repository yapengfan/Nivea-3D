"""Fit radial FK chains to the open gown, long tiered sleeves and hanging waist cords."""
import hashlib
from pathlib import Path
import numpy as np
from mathutils.bvhtree import BVHTree


BODY = "GPT_Body_Hair_Costume"
LEFT = "Dress | side cascade.L"
SKIRT_Z = np.array([.087, -.044, -.180, -.319, -.455])
SLEEVE_Z = np.array([.165, .034, -.111, -.267, -.413])
TAGS = tuple((i, side, sign * (i-.5)*np.pi/6)
             for side, sign in (("L",1),("R",-1)) for i in range(1,7))
LARIAT = {81,82,83,93,94}


def arrays(o):
    p=np.empty((len(o.data.vertices),3),dtype='f')
    o.data.vertices.foreach_get("co",p.ravel())
    mat=np.asarray(o.matrix_world)
    o.data.calc_loop_triangles()
    f=np.empty((len(o.data.loop_triangles),3),dtype='i')
    o.data.loop_triangles.foreach_get("vertices",f.ravel())
    return p@mat[:3,:3].T+mat[:3,3], f


def domains(f):
    with np.load(Path(__file__).with_name("cloth_regions.npz")) as d:
        if hashlib.sha256(f.astype('<i4').tobytes()).hexdigest()!=d["topology"].item():
            raise ValueError("Cloth segmentation does not match the base rig.")
        return d['regions']


def crest_center(z):
    # The long back train flares behind the legs, increasingly towards the hem.
    return np.array([np.interp(z,[-.46,-.30,-.14,.08],[.009,.075,.106,.130]),-.01222])


def slices(p,f,z):
    t=p[f]
    crossed=t[(t[:,:,2].max(1)>z)&(t[:,:,2].min(1)<=z)]
    samples=[]
    for tri in crossed:
        ends=[]
        for a,b in zip(tri,np.roll(tri,1,axis=0)):
            if (a[2]>z)!=(b[2]>z):
                ends.append(a+(b-a)*(z-a[2])/(b[2]-a[2]))
        if len(ends)==2:
            steps=max(2,int(np.linalg.norm(ends[0]-ends[1])/.004))
            samples.extend(np.linspace(*ends,steps))
    if not samples:
        ix=np.argsort(abs(p[:,2]-z))[:12]
        return p[ix]
    return np.asarray(samples)


def skirt_guide(ring,z,angle):
    radial=ring[:,:2]-crest_center(z)
    delta=np.abs(np.angle(np.exp(1j*(np.arctan2(radial[:,1],radial[:,0])-angle))))
    idx=np.argsort(delta)[:min(18,len(delta))]
    pos=np.median(ring[idx],axis=0)
    return pos


def prefix(index):
    return f"Skirt.{index:02d}."


def chain(out,stem,side,points,parent):
    for n,(a,b) in enumerate(zip(points,points[1:]),1):
        key=f"{stem}{n:02d}.{side}"
        out[key]=(a,b,parent)
        parent=key


def island_ids(o):
    a=o.data.attributes['Source_Island']
    ids=np.empty(len(o.data.vertices),'i')
    a.data.foreach_get('value',ids)
    return ids


def cord_specs(scene):
    for side,label,stem,ids,zvals in [
            ("L","Waist | left knot and lariat","WaistDrop.",LARIAT,(.030,-.015,-.041,-.095)),
            ("R","Waist | right knot and drops","WaistDrop.",{38,39},(.077,.040)),
            ("R","Waist | right knot and drops","WaistDrop.Inner.",{36,37},(.080,.043))]:
        obj=scene.objects[label]
        points=arrays(obj)[0]
        include=np.isin(island_ids(obj),list(ids))
        p=points[include]
        shaft=[]
        for z in zvals:
            center=np.mean(p[np.argsort(abs(p[:,2]-z))[:min(6,len(p))]],axis=0)
            center[2]=z
            shaft.append(center)
        yield obj,side,stem,shaft,np.flatnonzero(include)


def bones(scene):
    p,f=arrays(scene.objects[BODY])
    regs=domains(f)
    skirt=f[(regs[f]==1).all(1)]
    rings=[slices(p,skirt,z) for z in SKIRT_Z]
    out={}
    for i,side,angle in TAGS:
        pos=[skirt_guide(r,z,angle) for r,z in zip(rings,SKIRT_Z)]
        chain(out,prefix(i),side,pos,"Pelvis")
    for side in ("L","R"):
        if side=="L":
            pts,tris=arrays(scene.objects[LEFT])
        else:
            pts,tris=p,f[(regs[f]==6).all(1)]
        cross=[slices(pts,tris,z) for z in SLEEVE_Z]
        points=[np.median(c,axis=0) for c in cross]
        chain(out,"Sleeve.",side,points,"UpperArm."+side)
    for _,side,stem,pos,_ in cord_specs(scene):
        chain(out,stem,side,pos,"Pelvis")
    return out


def smooth(x):
    t=np.clip(x,0.,1.)
    return t*t*(3.-2.*t)


def along(z,levels,stem,side):
    centers=(levels[:-1]+levels[1:])*.5
    s=np.interp(-z,-centers,np.arange(len(centers)))
    n=int(np.floor(s))
    names={f"{stem}{n+1:02d}.{side}":1.-(s-n)}
    if s-n>0.:
        names[f"{stem}{n+2:02d}.{side}"]=s-n
    return names


def mix(row,source,t):
    result={n:v*(1.-t) for n,v in row.items()}
    for n,v in source.items():
        result[n]=result.get(n,0.)+v*t
    return result


def skirt_weights(v,base):
    xy=v[:2]-crest_center(v[2])
    theta=np.arctan2(xy[1],xy[0])
    frac=(theta/(np.pi/6)-.5)%12.
    q=int(np.floor(frac))
    phase=frac-q
    def key(k):
        k%=12
        return (k+1,"L") if k<6 else (12-k,"R")
    a,sa=key(q)
    b,sb=key(q+1)
    left=along(v[2],SKIRT_Z,prefix(a),sa)
    right=along(v[2],SKIRT_Z,prefix(b),sb)
    cloth=mix(left,right,float(smooth(phase)))
    # A small thigh share remains in the front panels so a first stride carries the hem.
    front=smooth((v[0]+.035)/.21)
    takeover=(.99-.14*front)*smooth((.099-v[2])/.128)
    return mix(base,cloth,float(takeover))


def sleeve_weights(v,side,base):
    effect=.975*smooth((.166-v[2])/.086)
    return mix(base,along(v[2],SLEEVE_Z,"Sleeve.",side),float(effect))


def get_weights(obj,rig,i):
    return {obj.vertex_groups[g.group].name:float(g.weight)
            for g in obj.data.vertices[i].groups if obj.vertex_groups[g.group].name in rig.data.bones}


def put(obj,rig,rows):
    vgs={g.name:g for g in obj.vertex_groups if g.name in rig.data.bones}
    ids=list(rows)
    for g in vgs.values(): g.remove(ids)
    for i,r in rows.items():
        top=sorted(((k,v) for k,v in r.items() if v>.0001),key=lambda kv:-kv[1])[:6]
        total=sum(v for _,v in top)
        if not total: raise ValueError(f"No garment weights on {obj.name} vertex {i}")
        for k,v in top:
            if k not in vgs:vgs[k]=obj.vertex_groups.new(name=k)
            vgs[k].add([i],float(v/total),'REPLACE')


def projected(tree,face,points,rows,v):
    p,_,i,_=tree.find_nearest(v)
    ids=face[i]
    a,b,c=points[ids]
    uv=np.linalg.lstsq(np.stack([b-a,c-a],axis=1),np.array(p)-a,rcond=None)[0]
    bary=np.clip([1.-uv.sum(),*uv],0.,1.)
    result={}
    for idx,factor in zip(ids,bary):
        for n,w in rows[int(idx)].items():
            result[n]=result.get(n,0.)+float(w*factor)
    return result


def apply(scene,rig,defs):
    main=scene.objects[BODY]
    p,f=arrays(main)
    reg=domains(f)
    idx=np.flatnonzero((reg==1)|(reg==6))
    main_rows={}
    for j in idx:
        j=int(j)
        previous=get_weights(main,rig,j)
        main_rows[j]=skirt_weights(p[j],previous) if reg[j]==1 else sleeve_weights(p[j],"R",previous)
    put(main,rig,main_rows)
    left=scene.objects[LEFT]
    lpoints,_=arrays(left)
    put(left,rig,{j:sleeve_weights(v,"L",get_weights(left,rig,j)) for j,v in enumerate(lpoints)})
    supports={}
    for region in (1,6):
        subset=f[(reg[f]==region).all(1)]
        tree=BVHTree.FromPolygons(p.tolist(),subset.tolist(),all_triangles=True)
        supports[region]=(tree,subset,p,main_rows)
    counts={main.name:len(idx),left.name:len(lpoints)}
    for name in ("Dress | lower hem motifs.R","Dress | diagonal hem motifs.L"):
        obj=scene.objects[name]
        points,_=arrays(obj)
        forms={j:projected(*supports[6 if name.endswith(".R") and v[2]>-.335 else 1],v)
               for j,v in enumerate(points)}
        put(obj,rig,forms)
        counts[name]=len(forms)
    for obj,side,stem,shaft,selection in cord_specs(scene):
        pts=arrays(obj)[0]
        z=np.array(shaft)[:,2]
        rows={}
        for i in selection:
            v=pts[i]
            t=smooth((z[0]-v[2])/(.012 if side=="L" else .009))
            rows[int(i)]=mix(get_weights(obj,rig,int(i)),along(v[2],z,stem,side),float(t))
        put(obj,rig,rows)
        counts[obj.name]=counts.get(obj.name,0)+len(rows)
    return {"vertices":counts,"skirt_chains":12,"skirt_segments":4,
            "sleeve_chains":2,"waist_chains":3}
