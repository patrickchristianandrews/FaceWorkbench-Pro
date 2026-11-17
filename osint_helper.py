import io
from PIL import Image,ExifTags
SITES={'Yandex Images':'https://yandex.com/images/','Lenso.ai':'https://lenso.ai/','PimEyes':'https://pimeyes.com/','FaceCheck.id':'https://facecheck.id/','Search4Faces':'https://search4faces.com/','FindClone':'https://findclone.ru/'}
def metadata(raw):
    with Image.open(io.BytesIO(raw)) as im:
        info={'format':im.format,'dimensions':list(im.size)}
        for k,v in im.getexif().items():
            label=ExifTags.TAGS.get(k,str(k))
            if label not in ('MakerNote','UserComment'): info[label]=str(v)[:500]
        return info
