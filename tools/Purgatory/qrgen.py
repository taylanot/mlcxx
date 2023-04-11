#import qrcode
#import qrcode.image.svg
#
#method='fragment'
#if method == 'basic':
#    # Simple factory, just a set of rects.
#    factory = qrcode.image.svg.SvgImage
#elif method == 'fragment':
#    # Fragment factory (also just a set of rects)
#    factory = qrcode.image.svg.SvgFragmentImage
#else:
#    # Combined path factory, fixes white space that may occur when zooming
#    factory = qrcode.image.svg.SvgPathImage
#
#qr = qrcode.QRCode(image_factory=factory)
#qr.add_data('https://github.com/taylanot/EE_MAML')
#qr.make(fit=True)
#img = qr.make_image(image_factory=factory,fill_color=(159, 93, 153))
##img = qr.make_image(fill_color=(159, 93, 153))

import qrcode
from qrcode.image.styles.moduledrawers.svg import SvgPathCircleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask
from qrcode.image.svg import SvgFragmentImage
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data('https://github.com/taylanot/EE_MAML')

img = qr.make_image(image_factory=SvgFragmentImage(), module_drawer=SvgPathCircleDrawer())

img.save("test.svg")
