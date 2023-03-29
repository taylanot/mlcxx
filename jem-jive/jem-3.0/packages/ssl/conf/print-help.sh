
cat <<EOF
  --with-ssl            : enable the ssl package.
  --without-ssl         : disable the ssl package.
                         (default: --with-ssl)

  --ssl-libs=LIBS       : use LIBS to link with the OpenSSL ssl
                          library.
  --ssl-libdirs=DIRS    : search in DIRS for the OpenSSL ssl
                          library.
  --ssl-incdirs=DIRS    : search in DIRS for the OpenSSL include
                          directory containing the ssl header files.
EOF
