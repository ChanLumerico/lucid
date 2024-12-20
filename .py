import lucid
import lucid.nn.functional as F


img = lucid.arange(7).reshape(-1, 1).repeat(7, axis=1)
img = img.reshape(1, 1, 7, 7)

rot = F.rotate(img, angle=45)
print(rot)
