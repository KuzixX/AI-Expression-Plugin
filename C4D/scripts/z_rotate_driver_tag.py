"""
Z-Rotate Driver — Python Tag для Cinema 4D.

Нуль двигается по своему локальному Z → два джойнта крутятся вокруг выбранной
локальной оси на угол, пропорциональный смещению нуля от исходной позиции:

    angle = (null.localZ - restZ) * DegPerUnit      # градусы

Знак Z задаёт сторону вращения (ушёл в минус → крутит в другую сторону).
Позиция джойнтов НЕ трогается — только ротация (не конфликтует с позиционными
драйверами). Нейтраль (rest Z нуля + нейтральные матрицы джойнтов) ловится при
первом запуске или по кнопке Recapture.

User Data (создаётся автоматически):
  1 Null (control)   — нуль-контроллер
  2 Joint A          — первый джойнт
  3 Joint B          — второй джойнт
  4 Axis (local)     — ось вращения: X / Y / Z (в локальном пространстве джойнта)
  5 Deg per unit     — градусов на единицу смещения Z (чувствительность)
  6 Recapture neutral— кнопка: перезахватить нейтраль (поставь всё в rest и тикни)
"""
import c4d
from c4d import utils

_neutral = {}   # id(joint) -> neutral local Matrix
_restz   = {}   # id(null)  -> rest local Z


def _link_bc(name):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_BASELISTLINK)
    bc[c4d.DESC_NAME] = name; bc[c4d.DESC_SHORT_NAME] = name
    return bc


def _real_bc(name, default):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_REAL)
    bc[c4d.DESC_NAME] = name; bc[c4d.DESC_SHORT_NAME] = name
    bc[c4d.DESC_DEFAULT] = default; bc[c4d.DESC_STEP] = 0.1
    return bc


def _axis_bc(name):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_LONG)
    bc[c4d.DESC_NAME] = name; bc[c4d.DESC_SHORT_NAME] = name
    bc[c4d.DESC_CUSTOMGUI] = c4d.CUSTOMGUI_CYCLE
    cyc = c4d.BaseContainer()
    cyc.SetString(0, "X"); cyc.SetString(1, "Y"); cyc.SetString(2, "Z")
    bc[c4d.DESC_CYCLE] = cyc
    return bc


def _bool_bc(name):
    bc = c4d.GetCustomDataTypeDefault(c4d.DTYPE_BOOL)
    bc[c4d.DESC_NAME] = name; bc[c4d.DESC_SHORT_NAME] = name
    bc[c4d.DESC_CUSTOMGUI] = c4d.CUSTOMGUI_BOOL
    return bc


def _build(tag):
    tag.AddUserData(_link_bc("Null (control)"))
    tag.AddUserData(_link_bc("Joint A"))
    tag.AddUserData(_link_bc("Joint B"))
    tag.AddUserData(_axis_bc("Axis (local)"))
    tag.AddUserData(_real_bc("Deg per unit", 10.0))
    tag.AddUserData(_bool_bc("Recapture neutral"))


def _ud(tag, i):
    try:
        return tag[c4d.ID_USERDATA, i]
    except Exception:
        return None


def _rotmat(axis, rad):
    if axis == 0: return utils.MatrixRotX(rad)
    if axis == 1: return utils.MatrixRotY(rad)
    return utils.MatrixRotZ(rad)


def main():
    tag = op
    if not tag.GetUserDataContainer():
        _build(tag); c4d.EventAdd(); return

    null = _ud(tag, 1); jA = _ud(tag, 2); jB = _ud(tag, 3)
    axis = _ud(tag, 4) or 0
    degU = _ud(tag, 5)
    if degU is None:
        degU = 0.0
    recap = _ud(tag, 6)
    if null is None:
        return

    joints = [j for j in (jA, jB) if j is not None]
    nkey = id(null)

    if recap or nkey not in _restz:
        _restz[nkey] = null.GetRelPos().z
        for j in joints:
            _neutral[id(j)] = j.GetMl()
        if recap:
            tag[c4d.ID_USERDATA, 6] = False
        return

    dz = null.GetRelPos().z - _restz[nkey]
    rot = _rotmat(int(axis), utils.Rad(dz * degU))

    for j in joints:
        neu = _neutral.get(id(j))
        if neu is None:
            _neutral[id(j)] = neu = j.GetMl()
        m = neu * rot                 # доворот вокруг локальной оси от нейтрали
        m.off = j.GetMl().off         # позицию не трогаем
        j.SetMl(m)
