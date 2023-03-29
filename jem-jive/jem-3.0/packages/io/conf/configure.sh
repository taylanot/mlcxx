
#========================================================================
#   Global variables
#========================================================================


USE_ZLIB=1
USE_READLINE=1
USE_GNU_READLINE=""

Z_LIBS="z"
Z_LIBDIRS=""
Z_INCDIRS=""

READLN_LIBS="readline"
READLN_LIBDIRS=""
READLN_INCDIRS=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   SetDefaults ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


SetDefaults()
{
  local f="$(FindSysconfFile defaults.sh)"

  if [ -n "$f" ]; then
    Log "Executing \`$f' ..."
    . "$f"
  fi
}


#------------------------------------------------------------------------
#   ParseArgs ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


ParseArgs()
{
  local arg=""

  for arg in "$@" ; do

    case "$arg" in

      --z-libs=?*)

        Z_LIBS="${arg#--z-libs=}" ;;

      --z-libdirs=?*)

        Z_LIBDIRS="${arg#--z-libdirs=}" ;;

      --z-incdirs=?*)

        Z_INCDIRS="${arg#--z-incdirs=}" ;;

      --with-zlib)

        USE_ZLIB=1 ;;

      --without-zlib)

        USE_ZLIB="" ;;

      --readln-libs=?*)

        READLN_LIBS="${arg#--readln-libs=}" ;;

      --readln-libdirs=?*)

        READLN_LIBDIRS="${arg#--readln-libdirs=}" ;;

      --readln-incdirs=?*)

        READLN_INCDIRS="${arg#--readln-incdirs=}" ;;

      --with-readline)

        USE_READLINE=1 ;;

      --without-readline)

        USE_READLINE="" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestZlib ()
#------------------------------------------------------------------------


TestZlib()
{
  local f="$(FindSysconfFile test-zlib.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$Z_LIBS"
  MY_LIBDIRS="$Z_LIBDIRS"
  MY_INCDIRS="$Z_INCDIRS"

  Log    "Testing zlib setup ..."
  printf "Testing if zlib works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    return 0
  else
    echo "no."
    return 1
  fi
}


#------------------------------------------------------------------------
#   TestReadline ()
#------------------------------------------------------------------------


TestReadline()
{
  local f="$(FindSysconfFile test-readline.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$READLN_LIBS"
  MY_LIBDIRS="$READLN_LIBDIRS"
  MY_INCDIRS="$READLN_INCDIRS"

  Log    "Testing readline setup ..."
  printf "Testing if readline works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
  else
    echo "no."
    return 1
  fi

  f="$(FindSysconfFile test-gnu-readline.cpp)"

  if [ -z "$f" ]; then
    return 0
  fi

  Log    "Checking for GNU readline ... "
  printf "Checking for GNU readline ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    USE_GNU_READLINE=1
  else
    echo "no."
    USE_GNU_READLINE=""
  fi

  return 0
}


#------------------------------------------------------------------------
#   PrintConfig()
#------------------------------------------------------------------------


PrintConfig()
{
  local f="$(FindConfigFile config.h)"

  cat<<EOF

// Generated on $(date).
// CHANGES TO THIS FILE WILL BE LOST.

#ifndef JEM_IO_CONFIG_H
#define JEM_IO_CONFIG_H

EOF

  if [ -n "$USE_ZLIB" ]; then
    echo "#define JEM_USE_ZLIB"
  fi

  if [ -n "$USE_READLINE" ]; then
    echo "#define JEM_USE_READLINE"
  fi

  if [ -n "$USE_GNU_READLINE" ]; then
    echo "#define JEM_USE_GNU_READLINE"
  fi

  if [ -n "$f" ]; then
    cat "$f"
  fi

  echo ""
  echo "#endif"
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_ZLIB" ] && TestZlib ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-zlib"
    echo "  --z-libs=$(quote "$Z_LIBS")"
    echo "  --z-libdirs=$(quote "$Z_LIBDIRS")"
    echo "  --z-incdirs=$(quote "$Z_INCDIRS")"
    echo ""

  fi

  PushDir src || Die

  cp -f native/ZlibInflator.h    "$PKG_INC_DIR/Inflator.h"   || Die
  cp -f native/ZlibDeflator.h    "$PKG_INC_DIR/Deflator.h"   || Die
  cp -f native/ZlibInflator.cpp   Inflator.cpp               || Die
  cp -f native/ZlibDeflator.cpp   Deflator.cpp               || Die

  PopDir || Die

else

  Z_LIBS=""
  Z_LIBDIRS=""
  Z_INCDIRS=""

  PushDir src || Die

  cp -f native/DummyInflator.h   "$PKG_INC_DIR/Inflator.h"   || Die
  cp -f native/DummyDeflator.h   "$PKG_INC_DIR/Deflator.h"   || Die
  cp -f native/DummyInflator.cpp  Inflator.cpp               || Die
  cp -f native/DummyDeflator.cpp  Deflator.cpp               || Die

  PopDir || Die

  echo "Disabled zlib support."

fi

if [ -n "$USE_READLINE" ] && TestReadline ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-readline"
    echo "  --readln-libs=$(quote "$READLN_LIBS")"
    echo "  --readln-libdirs=$(quote "$READLN_LIBDIRS")"
    echo "  --readln-incdirs=$(quote "$READLN_INCDIRS")"
    echo ""

  fi

  PushDir src || Die

  cp -f native/ReadlineTermReader.h \
        "$PKG_INC_DIR/StdTermReader.h"                   || Die
  cp -f native/ReadlineTermReader.cpp  StdTermReader.cpp || Die

  PopDir || Die

else

  READLN_LIBS=""
  READLN_LIBDIRS=""
  READLN_INCDIRS=""

  PushDir src || Die

  cp -f native/SimpleTermReader.h \
        "$PKG_INC_DIR/StdTermReader.h"                 || Die
  cp -f native/SimpleTermReader.cpp  StdTermReader.cpp || Die

  PopDir || Die

  echo "Disabled readline support."

fi

WriteFile PrintConfig "$PKG_INC_DIR/config.h"

MY_LIBS="$Z_LIBS $READLN_LIBS"
MY_LIBDIRS="$Z_LIBDIRS $READLN_LIBDIRS"
MY_INCDIRS="$Z_INCDIRS $READLN_INCDIRS"
