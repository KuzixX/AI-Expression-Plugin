#!/usr/bin/env python3
"""
Авто-постановка anchor-точек через MediaPipe Face Mesh.

Идея: рендерим голову ОРТОГРАФИЧЕСКИ спереди (растеризация через рейкаст —
shade по нормали грани), запускаем MediaPipe Face Mesh на этой картинке,
берём заранее выбранные индексы лендмарок, для каждого пускаем тот же
ортографический луч обратно в меш → точка попадания → ближайшая вершина = anchor.

Рендер и рейкаст используют ОДИН И ТОТ ЖЕ ортокадр, поэтому пиксель лендмарка и
луч строго согласованы (нет проблемы калибровки камеры).

MediaPipe Face Mesh: 468 лендмарок с фиксированной семантикой. Набор по
умолчанию (DEFAULT_LANDMARKS) — симметричные точки мышечных зон лица.
"""
import numpy as np


# MediaPipe Face Mesh — индексы лендмарок по умолчанию (canonical 468).
# 9 — переносица (верх носа), 4 — кончик носа, 199 — подбородок.
DEFAULT_LANDMARKS = [9, 4, 199]


def _shade_image(verts, faces, axis=2, sign=1.0, res=512, pad=0.06):
    """Ортографический рендер спереди через рейкаст: для каждого пикселя пускаем
    луч вдоль оси проекции, шейдим по нормали грани (диффуз к камере).

    axis: ось проекции (0=X,1=Y,2=Z). sign: с какой стороны камера (+1 → камера
    на +оси смотрит в −, обычно лицо вдоль +Z → axis=2, sign=+1).

    Возвращает (img uint8 HxWx3 RGB, ray_origins (res*res,3), ray_dir (3,),
    bbox2d (xmin,xmax,ymin,ymax по экранным осям), screen_axes (a0,a1))."""
    import open3d as o3d
    V = np.asarray(verts, dtype=np.float32)
    F = np.asarray(faces, dtype=np.uint32)
    # экранные оси: две оси, перпендикулярные axis проекции
    others = [i for i in range(3) if i != axis]
    a0, a1 = others                      # горизонталь, вертикаль экрана
    mn = V.min(0); mx = V.max(0)
    span = (mx - mn).max()
    p = pad * span
    x0, x1 = mn[a0] - p, mx[a0] + p
    y0, y1 = mn[a1] - p, mx[a1] + p
    depth0 = (mx[axis] + span) if sign > 0 else (mn[axis] - span)
    ddir = -1.0 if sign > 0 else 1.0     # луч идёт к голове

    # сетка лучей (res×res), пиксель [row,col], row 0 — верх (y=y1)
    us = np.linspace(x0, x1, res)
    vs = np.linspace(y1, y0, res)        # сверху вниз
    gu, gv = np.meshgrid(us, vs)
    origins = np.zeros((res * res, 3), dtype=np.float32)
    origins[:, a0] = gu.ravel()
    origins[:, a1] = gv.ravel()
    origins[:, axis] = depth0
    dirs = np.zeros((res * res, 3), dtype=np.float32)
    dirs[:, axis] = ddir

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V.astype(np.float64)),
                                  o3d.utility.Vector3iVector(faces.astype(np.int32)))))
    rays = np.concatenate([origins, np.broadcast_to(dirs, origins.shape)], axis=1)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy()
    nrm = ans['primitive_normals'].numpy()       # (N,3)
    hitmask = np.isfinite(t_hit)

    # diffuse shade по |нормаль·ось камеры| + ambient
    cam = np.zeros(3); cam[axis] = -ddir          # направление НА камеру
    sh = np.abs(nrm @ cam)
    val = (0.25 + 0.75 * sh)
    val[~hitmask] = 0.0
    img = (np.clip(val, 0, 1).reshape(res, res) * 255).astype(np.uint8)
    img_rgb = np.repeat(img[:, :, None], 3, axis=2)
    return (img_rgb, origins, dirs[0], (x0, x1, y0, y1), (a0, a1), axis,
            depth0, ddir, scene)


def _detect_landmarks(img_rgb):
    """MediaPipe Face Mesh на картинке. Возвращает (N,2) норм. координаты [0,1]
    (x вправо, y вниз) или None, если лицо не найдено."""
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.3)
    res = fm.process(img_rgb)
    fm.close()
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    return np.array([[lm.x, lm.y] for lm in lms], dtype=np.float64)


def auto_anchors(verts, faces, landmark_indices=None, res=512, views=None):
    """Авто-постановка anchor'ов на меше через MediaPipe.

    Перебираем направления взгляда (views), на каждом рендерим и пробуем
    задетектить лицо; берём вид с детекцией. Для выбранных индексов лендмарок
    пускаем ортолуч → ближайшая вершина. Дубли убираем.

    Возвращает (anchor_vertex_indices list, debug dict с картинкой/видом)."""
    from scipy.spatial import cKDTree
    if landmark_indices is None:
        landmark_indices = DEFAULT_LANDMARKS
    if views is None:
        # (axis, sign): лицо обычно вдоль +Z; пробуем оба знака Z, затем X
        views = [(2, 1.0), (2, -1.0), (0, 1.0), (0, -1.0)]

    tree = cKDTree(np.asarray(verts, dtype=np.float64))
    best = None
    for axis, sign in views:
        (img, origins, ddir_vec, (x0, x1, y0, y1), (a0, a1), ax,
         depth0, ddir, scene) = _shade_image(verts, faces, axis, sign, res)
        lms = _detect_landmarks(img)
        n = 0 if lms is None else len(lms)
        if best is None or n > best['n']:
            best = dict(n=n, lms=lms, img=img, axis=ax, a0=a0, a1=a1,
                        x0=x0, x1=x1, y0=y0, y1=y1, depth0=depth0,
                        ddir=ddir, scene=scene, sign=sign)
        if lms is not None:
            break                                  # нашли лицо — хватит

    if best is None or best['lms'] is None:
        return [], {'ok': False, 'reason': 'лицо не найдено MediaPipe'}

    import open3d as o3d
    lms = best['lms']
    a0, a1, ax = best['a0'], best['a1'], best['axis']
    x0, x1, y0, y1 = best['x0'], best['x1'], best['y0'], best['y1']
    chosen, seen = [], set()
    miss = 0
    for li in landmark_indices:
        if li >= len(lms):
            continue
        u, v = lms[li]                              # норм. [0,1], y вниз
        wx = x0 + u * (x1 - x0)
        wy = y1 - v * (y1 - y0)                     # экран сверху вниз
        org = np.zeros(3, dtype=np.float32)
        org[a0] = wx; org[a1] = wy; org[ax] = best['depth0']
        d = np.zeros(3, dtype=np.float32); d[ax] = best['ddir']
        ray = np.concatenate([org, d])[None, :]
        ans = best['scene'].cast_rays(
            o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32))
        t = float(ans['t_hit'].numpy()[0])
        if not np.isfinite(t):
            miss += 1
            continue
        hit = org + t * d
        _, vi = tree.query(np.asarray(hit, dtype=np.float64))
        vi = int(vi)
        if vi not in seen:
            seen.add(vi); chosen.append(vi)
    dbg = {'ok': True, 'img': best['img'], 'axis': ax, 'sign': best['sign'],
           'n_landmarks': best['n'], 'n_anchors': len(chosen), 'miss': miss}
    return chosen, dbg


if __name__ == "__main__":
    # smoke-тест на FLAME
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    import debug_head1_pipeline as pipe
    v_t, sd, faces = pipe.load_flame(pipe.FLAME_PKL)
    V = pipe.normalize_bbox(pipe.apply_betas(v_t, sd, {}))
    idx, dbg = auto_anchors(V, faces)
    print("debug:", {k: v for k, v in dbg.items() if k != 'img'})
    print("anchors:", idx)
