
#========================================================================
#   Global variables
#========================================================================


USE_GL=1
USE_GLFW=1
USE_GLUT=1
USE_OSMESA=""

GL_LIBS="GL"
GL_LIBDIRS=""
GL_INCDIRS=""

GLFW_H="glfw.h"
GLFW_LIB="glfw3"
GLUT_H="glut.h"
GLUT_LIB="glut"
OSMESA_LIB=""
DISPLAY="DummyDisplay"
OS_DISPLAY="OSDummyDisplay"
GL_LOADER=""


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

      --gl-libs=?*)

        GL_LIBS="${arg#--gl-libs=}" ;;

      --gl-libdirs=?*)

        GL_LIBDIRS="${arg#--gl-libdirs=}" ;;

      --gl-incdirs=?*)

        GL_INCDIRS="${arg#--gl-incdirs=}" ;;

      --with-glfw)

        USE_GLUT=""
        USE_GLFW=1 ;;

      --without-glfw)

        USE_GLFW="" ;;

      --with-glut)

        USE_GLFW=""
        USE_GLUT=1 ;;

      --without-glut)

        USE_GLUT="" ;;

      --with-osmesa)

        USE_OSMESA=1 ;;

      --without-osmesa)

        USE_OSMESA="" ;;

      --osmesa=?*)

        OSMESA_LIB="${arg#--osmesa=}" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestGL ( [-s|--silent] )
#------------------------------------------------------------------------


TestGL()
{
  local silent=""
  local f="$(FindSysconfFile test-gl.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  case "$1" in
    -s|--silent) silent=1
  esac

  MY_LIBS="$GL_LIBS"
  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

  Log "Testing GL setup ..."

  if [ -z "$silent" ]; then
    printf "Testing if OpenGL works ... "
  fi

  if TestSyslibs "$f" ; then
    if [ -z "$silent" ]; then
      echo "yes."
    fi
    return 0
  else
    if [ -z "$silent" ]; then
      echo "no."
    fi
    return 1
  fi
}


#------------------------------------------------------------------------
#   TestGLFW ( [-s|--silent] )
#------------------------------------------------------------------------


TestGLFW()
{
  local silent=""
  local f="$(FindSysconfFile test-glfw.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  case "$1" in
    -s|--silent) silent=1
  esac

  MY_LIBS="$GLFW_LIB $GL_LIBS"
  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

  Log "Testing GLFW setup ..."

  if [ -z "$silent" ]; then
    printf "Testing if GLFW works ... "
  fi

  if TestSyslibs "$f" ; then
    if [ -z "$silent" ]; then
      echo "yes."
    fi
    return 0
  else
    if [ -z "$silent" ]; then
      echo "no."
    fi
    return 1
  fi
}


#------------------------------------------------------------------------
#   TestGLUT ( [-s|--silent] )
#------------------------------------------------------------------------


TestGLUT()
{
  local silent=""
  local f="$(FindSysconfFile test-glut.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  case "$1" in
    -s|--silent) silent=1
  esac

  MY_LIBS="$GLUT_LIB $GL_LIBS"
  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

  Log "Testing GLUT setup ..."

  if [ -z "$silent" ]; then
    printf "Testing if GLUT works ... "
  fi

  if TestSyslibs "$f" ; then
    if [ -z "$silent" ]; then
      echo "yes."
    fi
    return 0
  else
    if [ -z "$silent" ]; then
      echo "no."
    fi
    return 1
  fi
}


#------------------------------------------------------------------------
#   TestFreeGLUT ( [-s|--silent] )
#------------------------------------------------------------------------


TestFreeGLUT()
{
  local silent=""
  local f="$(FindSysconfFile test-freeglut.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  case "$1" in
    -s|--silent) silent=1
  esac

  MY_LIBS="$GLUT_LIB $GL_LIBS"
  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

  Log "Testing FreeGLUT setup ..."

  if [ -z "$silent" ]; then
    printf "Testing for FreeGLUT ... "
  fi

  if TestSyslibs "$f" ; then
    if [ -z "$silent" ]; then
      echo "yes."
    fi
    return 0
  else
    if [ -z "$silent" ]; then
      echo "no."
    fi
    return 1
  fi
}


#------------------------------------------------------------------------
#   SelectOSMesa
#------------------------------------------------------------------------


SelectOSMesa()
{
  local f="$(FindSysconfFile test-osmesa.cpp)"
  local lib=""

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

  for lib in "$OSMESA_LIB" OSMesa OSMesa16 OSMesa32; do

    if [ -z "$lib" ]; then
      continue
    fi

    MY_LIBS="$lib $GL_LIBS"

    if TestSyslibs "$f" ; then

      OSMESA_LIB="$lib"
      OS_DISPLAY="OSMesaDisplay"
      GL_LIBS="$MY_LIBS"

      return 0

    fi

  done

  OSMESA_LIB=""

  return 1
}


#------------------------------------------------------------------------
#   SelectOSDisplay
#------------------------------------------------------------------------


SelectOSDisplay()
{
  local silent=""

  case "$1" in
    -s|--silent) silent=1
  esac

  Log "Testing off-screen rendering support ..."

  if [ -z "$silent" ]; then
    printf "Testing off-screen rendering support ... "
  fi

  if SelectOSMesa ; then

    if [ -z "$silent" ]; then
      echo "yes ($OSMESA_LIB)."
    fi

    return 0

  fi

  if [ -z "$silent" ]; then
    echo "no."
  fi

  USE_OSMESA=""

  return 1
}


#------------------------------------------------------------------------
#   ConfigureSource ()
#------------------------------------------------------------------------


ConfigureSource()
{
  local f=""

  PushDir "$PKG_DIR/src" || Die

  if   [ -n "$GLFW_H" ]; then
    cp -f "native/$GLFW_H" "glfw.h" || Die
  elif [ -n "$GLUT_H" ]; then
    cp -f "native/$GLUT_H" "glut.h" || Die
  fi

  if [ -n "$GL_LOADER" ]; then
    cp -f "native/${GL_LOADER}.h"   "ftable/NativeLoader.h"   || Die
    cp -f "native/${GL_LOADER}.cpp" "ftable/NativeLoader.cpp" || Die
  fi

  f="${DISPLAY}.cpp"

  if [ ! -e "native/$f" ]; then
    f="DummyDisplay.cpp"
  fi

  cp -f "native/$f" "$f" || Die

  f="${DISPLAY}.h"

  if [ -e "native/$f" ]; then
    cp -f "native/$f" "$f" || Die
  fi

  f="${OS_DISPLAY}.cpp"

  if [ ! -e "native/$f" ]; then
    f="OSDummyDisplay.cpp"
  fi

  cp -f "native/$f" "$f" || Die

  f="${OS_DISPLAY}.h"

  if [ -e "native/$f" ]; then
    cp -f "native/$f" "$f" || Die
  fi

  PopDir || Die
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_GL" ] && TestGL ; then

  if   [ -n "$USE_GLFW" ] && TestGLFW ; then

    GL_LIBS="$GLFW_LIB $GL_LIBS"
    DISPLAY="GlfwDisplay"
    USE_GLUT=""

  elif [ -n "$USE_GLUT" ] && TestGLUT ; then

    GL_LIBS="$GLUT_LIB $GL_LIBS"
    DISPLAY="GlutDisplay"
    USE_GLFW=""

    if TestFreeGLUT ; then
      GLUT_H=freeglut.h
    fi

  else

    USE_GLUT=""
    USE_GLFW=""

  fi

  if [ -z "$USE_GLUT" ]; then
    GLUT_H=""
  fi

  if [ -z "$USE_GLFW" ]; then
    GLFW_H=""
  fi

  if [ -n "$USE_OSMESA" ]; then
    SelectOSDisplay
  fi

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --gl-libs=$(quote "$GL_LIBS")"
    echo "  --gl-libdirs=$(quote "$GL_LIBDIRS")"
    echo "  --gl-incdirs=$(quote "$GL_INCDIRS")"

    if   [ -n "$USE_GLFW" ]; then
      echo "  --with-glfw"
    elif [ -n "$USE_GLUT" ]; then
      echo "  --with-glut"
    else
      echo "  --without-glut"
    fi

    if [ -n "$USE_OSMESA" ]; then
      echo "  --with-osmesa"
      echo "  --osmesa=$(quote "$OSMESA_LIB")"
    else
      echo "  --without-osmesa"
    fi

    echo ""

  fi

  ConfigureSource

  MY_LIBS="$GL_LIBS"
  MY_LIBDIRS="$GL_LIBDIRS"
  MY_INCDIRS="$GL_INCDIRS"

else

  LogEcho "Disabled the gl package."

  exit 0

fi
