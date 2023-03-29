
cat <<EOF
  --gl-libs=LIBS        : use LIBS to link with the OpenGL and
                          GLUT libraries.
  --gl-libdirs=DIRS     : search in DIRS for the OpenGL and
                          GLUT libraries.
  --gl-incdirs=DIRS     : search in DIRS for the OpenGL and
                          GLUT header files.

  --with-glut           : add support for the GLUT library.
  --without-glut        : disable support for the GLUT library.
                         (default: --with-glut)

  --with-glfw           : add support for the GLFW library.
                          Note that the GLFW library has priority over
                          the GLUT library; only one of them can be
                          enabled.
  --without-glfw        : disable support for the GLFW library.
                         (default: --with-glfw)

  --with-osmesa         : use the OSMesa (Off-Screen Mesa) library
                          to enable support for off-screen rendering.
  --without-osmesa      : do not use OSMesa.
                         (default: --without-osmesa)
  --osmesa=LIB          : use LIB to link with the OSMesa (Off-
                          Screen Mesa) library.
EOF
