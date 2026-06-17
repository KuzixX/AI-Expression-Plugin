"""
Мышечный риг: признаки мышц (позиция, направление, ID) и парные приоры
(geodesic vertex↔muscle, направленное выравнивание) для anatomical-bias
cross-attention.

ВАЖНО (заглушка/контракт): сейчас reference хранит только activations(M,).
Чтобы посчитать движение Σ_m W·act·dir, нужны на каждую мышцу:
  • origin/anchor point (3,)  — где мышца «закреплена» на голове,
  • direction (3,)            — куда тянет (для линейного скиннинга).
Эти данные приходят из твоего Unity-рига. Здесь — структуры + приоры, которые
из них вычисляются. Пока origin/direction можно задать заглушками, чтобы
обкатать пайплайн (см. make_dummy_rig).
"""
import numpy as np


class MuscleRig:
    """Описание мышечного рига (общее для всех голов в reference-кадре).
      names      — имена мышц (M,)
      origins    — точки крепления (M, 3)
      directions — единичные направления тяги (M, 3)
      radius     — радиус влияния (M,) для geodesic-приора
    """

    def __init__(self, names, origins, directions, radius=None):
        self.names = list(names)
        self.M = len(self.names)
        self.origins = np.asarray(origins, np.float32).reshape(self.M, 3)
        d = np.asarray(directions, np.float32).reshape(self.M, 3)
        self.directions = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        self.radius = (np.asarray(radius, np.float32) if radius is not None
                       else np.full(self.M, 0.15, np.float32))

    def muscle_features(self):
        """Признаки мышц для Muscle MLP: [origin(3), direction(3), radius(1)]
        = (M, 7). ID-эмбеддинг добавляется в модели по индексу столбца."""
        return np.concatenate(
            [self.origins, self.directions, self.radius[:, None]],
            axis=1).astype(np.float32)

    def pair_priors(self, verts):
        """Парные приоры (N, M): евклидово приближение geodesic (расстояние
        вершина→origin) и направленное выравнивание (cos между направлением
        мышцы и направлением origin→vertex).

        ПРИМ.: настоящий geodesic считается по мешу (potpourri3d heat method) —
        тут евклидов прокси для скорости/обкатки; заменяемо."""
        verts = np.asarray(verts, np.float32)
        diff = verts[:, None, :] - self.origins[None, :, :]   # (N, M, 3)
        dist = np.linalg.norm(diff, axis=2)                   # (N, M)
        geo = dist / (self.radius[None, :] + 1e-9)            # нормир. на радиус
        dirn = diff / (dist[:, :, None] + 1e-9)
        align = (dirn * self.directions[None, :, :]).sum(2)   # (N, M) cos
        return geo.astype(np.float32), align.astype(np.float32)


def make_dummy_rig(n_muscles, neutral_verts, seed=0):
    """Заглушка рига для обкатки пайплайна без реальных данных из Unity:
    origins — случайные вершины нейтрали, directions — случайные единичные."""
    rng = np.random.default_rng(seed)
    V = np.asarray(neutral_verts, np.float32)
    idx = rng.choice(len(V), size=n_muscles, replace=False)
    origins = V[idx]
    directions = rng.standard_normal((n_muscles, 3)).astype(np.float32)
    return MuscleRig([f"muscle_{i:02d}" for i in range(n_muscles)],
                     origins, directions)
