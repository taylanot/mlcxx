
PushDir "$PKG_DIR/src" || Die

if [ ! -e "$PKG_INC_DIR/native" ]; then
  MakeDir "$PKG_INC_DIR/native" || Die
fi

# Configure threads.

if [ -z "$USE_THREADS" ]; then

  cp -f native/DummyThread.h    "$PKG_INC_DIR/native/Thread.h"  || Die
  cp -f native/DummyThread.cpp   NativeThread.cpp               || Die
  cp -f native/DummyMonitor.h   "$PKG_INC_DIR/native/Monitor.h" || Die
  cp -f native/DummyMonitor.cpp  NativeMonitor.cpp              || Die

else

  cp -f native/PosixThread.h    "$PKG_INC_DIR/native/Thread.h"  || Die
  cp -f native/PosixThread.cpp   NativeThread.cpp               || Die
  cp -f native/PosixMonitor.h   "$PKG_INC_DIR/native/Monitor.h" || Die
  cp -f native/PosixMonitor.cpp  NativeMonitor.cpp              || Die

fi

# Configure System and Time.

cp -f native/PosixSystem.h      "$PKG_INC_DIR/native/System.h"  || Die
cp -f native/PosixSystem.cpp     NativeSystem.cpp               || Die
cp -f native/PosixTime.h        "$PKG_INC_DIR/native/Time.h"    || Die
cp -f native/PosixTime.cpp       NativeTime.cpp                 || Die
cp -f native/PosixProcess.h     "$PKG_INC_DIR/native/Process.h" || Die
cp -f native/PosixProcess.cpp    NativeProcess.cpp              || Die

PopDir || Die
