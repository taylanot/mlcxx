
. "${JEMDIR:?}/packages/gl/conf/print-help.sh"

echo ""

cat <<EOF
  --with-mesa           : use the Mesa OpenGL and GLUT libraries.
  --without-mesa        : use the native Apple OpenGL and GLUT
                          libraries.
                         (default: --without-mesa)
EOF
