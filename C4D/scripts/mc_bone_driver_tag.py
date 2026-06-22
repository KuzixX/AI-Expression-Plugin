"""
MC Bone Driver — Python Tag для Cinema 4D 2023.

Тег вешается на объект (например, родительский меш Male_01) и двигает КОСТЬ
в диапазоне от Neutral до Target по активации. У одной кости может быть
несколько таргетов — они складываются (additive), как в Unity SBonePositionDriver:

    current.pos = neutral.pos + Σ (target_i.pos - neutral.pos) * activation_i * weight_i

Поля (User Data) создаются автоматически при первом запуске тега:
  • Neutral        — объект-ориентир нейтральной (rest) позиции кости
  • Current (Bone) — кость, которую двигаем (если пусто — берётся объект тега)
  • Target N       — ориентир целевой позиции
  • Activation N   — 0..1, насколько кость уехала к этому таргету (это и анимируем)
  • Weight N       — множитель вклада таргета (по умолчанию 1)

Дублируй тег и раскидывай по костям — у каждого свои ссылки/активации.

ВАЖНО: Neutral и Target должны быть НЕЗАВИСИМЫМИ нулями (не потомками кости),
иначе при движении кости они поедут вместе с ней и получится обратная связь.
"""
import c4d

NUM_TARGETS = 6  # слотов таргетов на тег (хватает; при желании увеличь)


# ── построение User Data (один раз) ──────────────────────────────────────────
def _link_bc(name):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_BASELISTLINK)
    bc[c4d.DESC_NAME] = name
    bc[c4d.DESC_SHORT_NAME] = name
    return bc


def _real_bc(name, default, mn=None, mx=None, step=0.01):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_REAL)
    bc[c4d.DESC_NAME] = name
    bc[c4d.DESC_SHORT_NAME] = name
    bc[c4d.DESC_DEFAULT] = default
    bc[c4d.DESC_STEP] = step
    if mn is not None:
        bc[c4d.DESC_MIN] = mn
        bc[c4d.DESC_MINSLIDER] = mn
    if mx is not None:
        bc[c4d.DESC_MAX] = mx
        bc[c4d.DESC_MAXSLIDER] = mx
    return bc


def _build_userdata(tag):
    """Создаёт поля строго по порядку → детерминированные ID:
       1=Neutral, 2=Current, далее по 3 на таргет (Target/Activation/Weight)."""
    tag.AddUserData(_link_bc("Neutral"))            # id 1
    tag.AddUserData(_link_bc("Current (Bone)"))     # id 2
    for i in range(1, NUM_TARGETS + 1):
        tag.AddUserData(_link_bc("Target %d" % i))                       # 3,6,9,...
        tag.AddUserData(_real_bc("Activation %d" % i, 0.0, 0.0, 1.0))    # 4,7,10,...
        tag.AddUserData(_real_bc("Weight %d" % i, 1.0))                  # 5,8,11,...


def _ud(tag, idx):
    """Безопасное чтение значения User Data по числовому id."""
    try:
        return tag[c4d.ID_USERDATA, idx]
    except Exception:
        return None


# ── основной проход (каждый кадр) ────────────────────────────────────────────
def main():
    tag = op

    # первый запуск — построить поля и выйти (на след. проходе уже работаем)
    if not tag.GetUserDataContainer():
        _build_userdata(tag)
        c4d.EventAdd()
        return

    neutral = _ud(tag, 1)
    current = _ud(tag, 2)
    if current is None:
        current = tag.GetObject()        # фолбэк: объект, на котором висит тег
    if neutral is None or current is None:
        return

    base = neutral.GetMg().off
    offset = c4d.Vector(0.0)

    for k in range(NUM_TARGETS):
        tid = 3 + k * 3
        aid = 4 + k * 3
        wid = 5 + k * 3
        target = _ud(tag, tid)
        if target is None:
            continue
        act = _ud(tag, aid) or 0.0
        w = _ud(tag, wid)
        if w is None:
            w = 1.0
        offset += (target.GetMg().off - base) * (act * w)

    mg = current.GetMg()
    mg.off = base + offset               # двигаем только позицию, поворот не трогаем
    current.SetMg(mg)
