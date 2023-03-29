
PushDir "$PKG_DIR/src" || Die

cp -f native/NativeAddress.h      NativeAddress.h    || Die
cp -f native/NativeAddress.cpp    NativeAddress.cpp  || Die

if [ "$SYS_NAME" = "windows" ]; then
  cp -f native/WinSockLib.h       WinSockLib.h       || Die
  cp -f native/WinSockLib.cpp     WinSockLib.cpp     || Die
  cp -f native/WinHandle.h        NativeHandle.h     || Die
  cp -f native/WinHandle.cpp      NativeHandle.cpp   || Die
  cp -f native/WinSocket.h        NativeSocket.h     || Die
  cp -f native/WinSocket.cpp      NativeSocket.cpp   || Die
  cp -f native/WinServer.h        NativeServer.h     || Die
  cp -f native/WinServer.cpp      NativeServer.cpp   || Die
else
  cp -f native/PosixHandle.h      NativeHandle.h     || Die
  cp -f native/PosixHandle.cpp    NativeHandle.cpp   || Die
  cp -f native/PosixSocket.h      NativeSocket.h     || Die
  cp -f native/PosixSocket.cpp    NativeSocket.cpp   || Die
  cp -f native/PosixServer.h      NativeServer.h     || Die
  cp -f native/PosixServer.cpp    NativeServer.cpp   || Die
fi

PopDir || Die

