
cat <<EOF
  --with-zlib           : use the zlib data compression library.
  --without-zlib        : do not use zlib.
                         (default: --with-zlib)

  --z-libs=LIBS         : use LIBS to link with zlib.
  --z-libdirs=DIRS      : search in DIRS for zlib.
  --z-incdirs=DIRS      : search in DIRS for the zlib header files.

  --with-readline       : use the readline library for command line
                          editing.
  --without-readline    : do not use readline.
                         (default: --with-readline)

  --readln-libs=LIBS    : use LIBS to link with readline.
  --readln-libdirs=DIRS : search in DIRS for the readline libraries.
  --readln-incdirs=DIRS : search in DIRS for the readline header files.
EOF
