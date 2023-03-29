
rm -f "$PKG_INC_DIR/config.h" \
      "$PKG_INC_DIR/config.h.bak" \
      "$PKG_INC_DIR/mpi.h"

if [ -e src/mpi/Makefile ]; then
  rm -f src/mpi/NoMakefile
  mv    src/mpi/Makefile src/mpi/NoMakefile
fi
