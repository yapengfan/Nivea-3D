"""FK skin of the curved dragon tail in the refined rig's original rest space."""
from pathlib import Path
import numpy as np
from mathutils import Vector


NAMES = tuple(f'Tail.{i:02d}' for i in range(1,13))


def _sections():
    # Sections are zero-to-one equipotentials on the actual connected tail skin:
    # the 18-vertex socket is the inlet, the single distal apex the outlet.
    # Arc length parameterizes the fitted, section-centroid spline.
    with np.load(Path(__file__).with_name('tail_sections.npz')) as data:
        return data['joints'], data['stations'], data['coordinate']


def bones(scene):
    joints, _, coord = _sections()
    if len(scene.objects['Tail | scales'].data.vertices) != len(coord):
        raise ValueError('Tail sections require the refined dragon tail.')
    return {name:(tuple(a),tuple(b),parent) for name,a,b,parent in
            zip(NAMES,joints,joints[1:],('Pelvis',*NAMES[:-1]))}


def roll_hints(scene):
    joints,_,_ = _sections()
    previous = None
    normal = Vector((0,0,1))
    result = {}
    for name,a,b in zip(NAMES,joints,joints[1:]):
        direction = Vector(b-a).normalized()
        if previous is not None:
            normal = previous.rotation_difference(direction) @ normal
        normal = (normal - direction*normal.dot(direction)).normalized()
        result[name] = tuple(normal)
        previous = direction
    return result


def apply(scene,rig,defs):
    obj = scene.objects['Tail | scales']
    if not any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):
        raise ValueError('Missing existing tail armature modifier')
    _,station,coord = _sections()
    groups = [obj.vertex_groups.get(n) or obj.vertex_groups.new(name=n) for n in NAMES]
    pelvis=obj.vertex_groups['Pelvis']
    owned=(*groups,pelvis)
    vertices=list(range(len(coord)))
    for group in owned:
        group.remove(vertices)
    # The continuous cross-section coordinate avoids influences hopping from the
    # coiled apex to a spatially nearby, but distant, segment of the same tail.
    s = np.interp(coord, station, np.arange(len(station))) - .5
    anchor = np.clip(coord/.071,0.,1.)
    anchor *= anchor*(3.-2.*anchor)
    for i,(h,free) in enumerate(zip(s,anchor)):
        lower=int(np.floor(h)); t=h-lower
        weights=(((1.-t)**3)/6., (4.-6.*t*t+3.*t**3)/6.,
                 (1.+3.*t+3.*t*t-3.*t**3)/6., t**3/6.)
        row={}
        for j,weight in zip(range(lower-1,lower+3),weights):
            index=min(max(j,0),len(groups)-1)
            row[index]=row.get(index,0.) + free*weight
        if free<1:row[-1]=1.-free
        row={k:v for k,v in row.items() if v>1.e-5}
        total=sum(row.values())
        for k,value in row.items():
            group=pelvis if k==-1 else groups[k]
            group.add([i],float(value/total),'REPLACE')
    return {'vertices':len(coord),'bone_count':len(NAMES),'centerline_length':float(station[-1]),
            'socket_vertices':int(np.sum(coord<1.e-8)), 'bone_chain':list(NAMES)}
