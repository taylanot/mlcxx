
PushDir "$PKG_DIR/src" || Die

if [ ! -e "$PKG_INC_DIR/native" ]; then
  MakeDir "$PKG_INC_DIR/native" || Die
fi

cp -f native/winclean.h  "$PKG_INC_DIR/native/winclean.h"

# Configure threads.

if [ -z "$USE_THREADS" ]; then

  cp -f native/DummyThread.h    "$PKG_INC_DIR/native/Thread.h"  || Die
  cp -f native/DummyThread.cpp   NativeThread.cpp               || Die
  cp -f native/DummyMonitor.h   "$PKG_INC_DIR/native/Monitor.h" || Die
  cp -f native/DummyMonitor.cpp  NativeMonitor.cpp              || Die

else

  cp -f native/WinThread.h      "$PKG_INC_DIR/native/Thread.h"  || Die
  cp -f native/WinThread.cpp     NativeThread.cpp               || Die
  cp -f native/WinMonitor.h     "$PKG_INC_DIR/native/Monitor.h" || Die
  cp -f native/WinMonitor.cpp    NativeMonitor.cpp              || Die

fi

# Configure System and Time.

cp -f native/WinSystem.h        "$PKG_INC_DIR/native/System.h"  || Die
cp -f native/WinSystem.cpp       NativeSystem.cpp               || Die
cp -f native/WinTime.h          "$PKG_INC_DIR/native/Time.h"    || Die
cp -f native/WinTime.cpp         NativeTime.cpp                 || Die
cp -f native/WinProcess.h       "$PKG_INC_DIR/native/Process.h" || Die
cp -f native/WinProcess.cpp      NativeProcess.cpp              || Die

PopDir || Die
