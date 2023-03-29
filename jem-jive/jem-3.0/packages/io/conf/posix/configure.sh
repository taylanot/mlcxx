
. "${PKG_DIR:?}/conf/configure.sh"

if [ ! -e "$PKG_INC_DIR/native" ]; then
  MakeDir "$PKG_INC_DIR/native" || Die
fi

PushDir "$PKG_DIR/src" || Die

cp -f native/PosixFile.h \
      "$PKG_INC_DIR/native/File.h"                     || Die
cp -f native/PosixFile.cpp        NativeFile.cpp       || Die
cp -f native/PosixFileStream.h \
      "$PKG_INC_DIR/native/FileStream.h"               || Die
cp -f native/PosixFileStream.cpp  NativeFileStream.cpp || Die
cp -f native/PosixDirListing.h \
      "$PKG_INC_DIR/native/DirListing.h"               || Die
cp -f native/PosixDirListing.cpp  NativeDirListing.cpp || Die

PopDir || Die
