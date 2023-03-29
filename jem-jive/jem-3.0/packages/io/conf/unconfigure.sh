
rm -f     "$PKG_INC_DIR/StdTermReader.h" \
          "$PKG_INC_DIR/Inflator.h" \
          "$PKG_INC_DIR/Deflator.h" \
          "$PKG_INC_DIR/config.h" \
          "$PKG_INC_DIR/config.h.bak" \
          src/NativeFile.cpp \
          src/NativeFileStream.cpp \
          src/NativeDirListing.cpp \
          src/StdTermReader.cpp \
          src/Inflator.cpp \
          src/Deflator.cpp

rm -f -r "$PKG_INC_DIR/native"
