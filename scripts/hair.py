"""Trace the existing long locks on the connected head shell and skin their surfaces."""
from heapq import heappop, heappush
import numpy as np

HEAD = "Head and hair | connected shell"
_cache = {}


def components(adj, mask):
    todo=set(np.flatnonzero(mask))
    pieces=[]
    while todo:
        seq=[todo.pop()]
        for i in seq:
            near=[j for j,_ in adj[i] if j in todo]
            todo.difference_update(near)
            seq.extend(near)
        pieces.append(np.asarray(seq))
    return sorted(pieces,key=len,reverse=True)


def walk(adj, seeds, region):
    dist=np.full(len(adj),np.inf)
    parent=np.full(len(adj),-1,int)
    queue=[]
    for i in seeds:
        dist[i]=0.
        heappush(queue,(0.,i))
    while queue:
        cost,i=heappop(queue)
        if cost>dist[i]:continue
        for j,w in adj[i]:
            val=cost+w
            if region[j] and val<dist[j]:
                dist[j]=val
                parent[j]=i
                heappush(queue,(val,j))
    return dist,parent


def smooth(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def trace(p, adj, mask, end_ids, root):
    # Walk inside this hair surface. Nearby curls can overlap in space.
    tip=end_ids[np.argmin(p[end_ids,2])]
    dist,previous=walk(adj,[tip],mask)
    roots=np.where(mask & (p[:,2]>.325) & np.isfinite(dist))[0]
    start=roots[np.argmin(np.linalg.norm(p[roots]-root,axis=1)+.075*dist[roots])]
    path=[start]
    while path[-1]!=tip:
        path.append(previous[path[-1]])
    path=np.asarray(path)
    along=dist[start]-dist[path]
    dpath,_=walk(adj,path,mask)
    knots=[]
    for arclen in np.linspace(0.,along[-1],6):
        guide=np.array([np.interp(arclen,along,p[path,axis]) for axis in range(3)])
        # Average across the section of the same lock to place the joint inside it.
        radial=np.linalg.norm(p-guide,axis=1)
        local=mask & (radial<.028) & (dpath<.042)
        weight=np.exp(-(radial[local]/.017)**2-((p[local,2]-guide[2])/.011)**2)
        centroid=np.average(p[local],axis=0,weights=weight)
        knots.append(guide*.35+centroid*.65)
    return np.asarray(knots),dpath


def nearest_segment(p,knots):
    d=knots[1:]-knots[:-1]
    delta=p[:,None,:]-knots[:-1]
    t=np.clip(np.sum(delta*d,axis=2)/np.square(d).sum(axis=1),0,1)
    r=np.linalg.norm(delta-t[:,:,None]*d,axis=2)
    seg=np.argmin(r,axis=1)
    ids=np.arange(len(p))
    return seg+t[ids,seg],r[ids,seg]


def fringe_tips(p,adj,defs,skin):
    # These four terminal forks were registered on the V1 head cage. At the
    # indicated section each is disjoint from the skin and both eye regions.
    # Keep the ring bordering the scalp stationary, even on long triangles.
    guides=(("Bangs","R",1004,.380),("Bangs","L",6162,.395),
            ("BangsCenter","",1525,.385),("BangsInner","L",6534,.408))
    for stem,side,tip,height in guides:
        part=next(c for c in components(adj,p[:,2]<height) if tip in c)
        if not 10<=len(part)<=60:
            raise ValueError("The reference fringe fork no longer matches "+stem)
        mask=np.zeros(len(p),bool)
        mask[part]=True
        rim=[v for v in part if any(not mask[n] for n,_ in adj[v])]
        to_root,_=walk(adj,rim,mask)
        to_tip,_=walk(adj,[tip],mask)
        arclen=to_root[part]/np.maximum(to_root[part]+to_tip[part],1e-10)
        path=[p[rim].mean(0),np.average(p[part],axis=0,weights=np.exp(-((arclen-.5)/.21)**2)),p[tip]]
        suffix="."+side if side else ""
        keys=[stem+".01"+suffix,stem+".02"+suffix]
        for i,b in enumerate(keys):defs[b]=(tuple(path[i]),tuple(path[i+1]),keys[i-1] if i else 'Head')
        for v,t in zip(part,arclen):
            release=float(smooth(t/.4))
            bend=float(smooth((t-.3)/.44))
            row={'Head':1-release,keys[0]:release*(1-bend),keys[1]:release*bend}
            skin[int(v)]={b:w for b,w in row.items() if w>1e-8}


def fit(scene):
    o=scene.objects[HEAD]
    key=o.data.as_pointer()
    if key in _cache:return _cache[key]
    m=o.data
    mat=np.asarray(o.matrix_world)
    p=np.asarray([v.co[:] for v in m.vertices])@mat[:3,:3].T+mat[:3,3]
    edges=np.empty((len(m.edges),2),dtype='i')
    m.edges.foreach_get('vertices',edges.ravel())
    adj=[[] for v in p]
    for (i,j),d in zip(edges,np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)):
        adj[i].append((j,d));adj[j].append((i,d))
    pieces=components(adj,p[:,2]<.36)
    rear=next(c for c in pieces if len(c)>4000 and p[c,2].min()<-.035)
    front=[c for c in pieces if 180<len(c)<500 and p[c,0].min()>.105 and p[c,2].min()<.23]
    if len(front)!=2:raise ValueError("The separated face-framing locks could not be identified")
    lower=components(adj,p[:,2]<.07)
    # Five major families of rear curls per side, each represented by an authored terminal curl.
    labels=("HairTemple","HairOuter","HairBackOuter","HairBackMid","HairBackInner")
    families=[((-.038,.200),(.083,.061)),((-.061,.150),(.06,.071)),
              ((-.117,.085),(.036,.051)),((-.092,.054),(.019,.033)),
              ((-.105,.024),(.011,.008))]
    yc=-.013
    defs={}
    skin={}
    def solve(region,locks):
        mask=np.full(len(p),False)
        mask[region]=True
        keys=[];cost=[];phase=[];root_heights=[]
        for name,side,end,target in locks:
            knots,geod=trace(p,adj,mask,end,np.asarray(target))
            names=[f"{name}.{i:02d}.{side}" for i in range(1,6)]
            for i,b in enumerate(names):
                defs[b]=(tuple(knots[i]),tuple(knots[i+1]),names[i-1] if i else 'Head')
            along,euclid=nearest_segment(p[region],knots)
            cost.append(euclid*.82 + np.minimum(geod[region],.20)*.18)
            phase.append(along)
            keys.append(names)
            root_heights.append(knots[0,2])
        dist=np.asarray(cost).T
        # Soft attribution at merged roots; keep a continuous strand together near its tip.
        assign=np.exp(-np.square((dist-dist.min(axis=1,keepdims=True))/.025))
        assign/=assign.sum(1,keepdims=True)
        ix=np.full(len(p),-1,int);ix[region]=np.arange(len(region))
        if len(locks)>1:
            for piece in components(adj, mask & (p[:,2]<.158)):
                if len(piece)<5 or np.ptp(p[piece,1])>.112:continue
                ind=ix[piece]
                low=smooth((.158-p[piece,2])/.066)[:,None]
                coherent=assign[ind].mean(axis=0)
                assign[ind]=assign[ind]*(1-low)+coherent*low
        for local,v in enumerate(region):
            mixture={}
            two=np.argsort(assign[local])[-2:]
            total=assign[local,two].sum()
            release=float(smooth((.360-p[v,2])/.048))
            for chain in two:
                strength=release*assign[local,chain]/total
                t=np.clip(phase[chain][local]-.5,0.,4.)
                base=int(t)
                frac=float(smooth(t-base))
                for bone,val in ((keys[chain][base],strength*(1-frac)),
                                 (keys[chain][min(4,base+1)],strength*frac)):
                    if val>1e-6:mixture[bone]=mixture.get(bone,0.)+float(val)
            mixture['Head']=1.-release
            sparse=sorted(((k,w) for k,w in mixture.items() if w>1e-6),key=lambda item:-item[1])[:4]
            total=sum(w for _,w in sparse)
            skin[int(v)]={k:w/total for k,w in sparse}
    rear_locks=[]
    for side,s in [('L',1),('R',-1)]:
        sy=lambda y:yc+s*(y-yc)
        for name,((x,y),(xr,yr)) in zip(labels,families):
            terminal=min((c for c in lower if len(c)>25),key=lambda c:np.linalg.norm(p[c,:2].mean(0)-[x,sy(y)]))
            rear_locks.append((name,side,terminal,(xr,sy(yr),.347)))
    solve(rear,rear_locks)
    for part in front:
        side="L" if p[part,1].mean()>yc else "R"
        top=part[p[part,2]>.343]
        solve(part,[("HairFace",side,part,np.mean(p[top],0))])
    fringe_tips(p,adj,defs,skin)
    # The scanned back hair shares a few vertices with the restored face and neck.
    # Read the actual retouch UV mask and the existing neck weighting, then fade
    # out the new influences geodesically before either protected region.
    mapped=m.uv_layers.get("Face pigment | local projection")
    protected=set()
    if mapped:
        loop_uv=np.empty((len(m.loops),2),np.float32)
        mapped.uv.foreach_get('vector',loop_uv.ravel())
        for poly in m.polygons:
            if not np.all(loop_uv[poly.loop_start]==-2.):protected.update(poly.vertices)
    base_groups={g.index:g.name for g in o.vertex_groups}
    for v in m.vertices:
        if any(base_groups[g.group]=='Neck' and g.weight>.005 for g in v.groups):
            protected.add(v.index)
    if protected:
        distance,_=walk(adj,protected,np.ones(len(p),bool))
        for idx in list(skin):
            if distance[idx]<=1e-8:
                del skin[idx]
            elif distance[idx]<.023:
                mix=float(smooth(distance[idx]/.023))
                row={b:w*mix for b,w in skin[idx].items()}
                for g in m.vertices[idx].groups:
                    b=base_groups[g.group]
                    row[b]=row.get(b,0.)+(1-mix)*g.weight
                keep=sorted(row.items(),key=lambda pair:-pair[1])[:4]
                total=sum(w for _,w in keep)
                skin[idx]={b:w/total for b,w in keep}
    _cache[key]=defs,skin
    return defs,skin


def bones(scene):
    return fit(scene)[0]


def apply(scene,rig,defs):
    defs,skin=fit(scene)
    obj=scene.objects[HEAD]
    active=list(skin)
    for group in obj.vertex_groups:
        if rig.data.bones.get(group.name):
            group.remove(active)
    names={bone for row in skin.values() for bone in row}
    groups={name:obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            for name in sorted(names)}
    for idx,row in skin.items():
        for name,w in row.items():groups[name].add([idx],w,'REPLACE')
    return dict(long_chains=12,fringe_chains=4,bones=len(defs),vertices=len(skin),source_reference="12 five-link scalp chains and five short fringe chains",
                facial_shell="Original base-rig weights on the face and scalp")
