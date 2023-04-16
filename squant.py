def creation(orbital, determinant):
  """takes the orbital being added and the determinant being acted upon.
    returns the phase factor, if the orbital being added is not in the determinant yet."""
  face_factor=-1
  if orbital in determinant:
    return 0
  if orbital not in determinant:
    determinant.add(orbital)
  return face_factor**orbital
def annihalation(orbital, determinant):
  """takes the orbital being removed and the determinant being acted upon.
    returns the phase factor, if the orbital being removed is in the determinant."""
  face_factor=-1
  if orbital not in determinant:
    return 0
  if orbital in determinant:
    determinant.remove(orbital)
    face_factor**orbital
    


