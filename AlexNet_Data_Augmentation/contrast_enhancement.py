from PIL import ImageEnhance
class ContrastEnhancement(object):
    def __init__(self, factor=1.5):
         self.factor = factor

    def __call__(self, img):
         enhancer = ImageEnhance.Contrast(img)
         img = enhancer.enhance(self.factor)
         return img
