
cat <<EOF
  --with-crypto         : enable the crypto package.
  --without-crypto      : disable the crypto package.
                         (default: --with-crypto)

  --crypto-libs=LIBS    : use LIBS to link with the OpenSSL crypto
                          library.
  --crypto-libdirs=DIRS : search in DIRS for the OpenSSL crypto
                          library.
  --crypto-incdirs=DIRS : search in DIRS for the OpenSSL include
                          directory containing the crypto header files.
EOF
