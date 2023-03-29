
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   SetLinuxGLDefaults ()
#------------------------------------------------------------------------


SetLinuxGLDefaults()
{
  local x11root=""
  local dir=""

  GL_LIBS="GLU GL X11"
  GLFW_LIB="glfw"
  GL_LOADER="GLXLoader"

  if ! FindWord --silent "pthread" $SYS_LIBS; then
    GL_LIBS="$GL_LIBS pthread"
  fi

  if   [ -d /usr/X11/lib ]; then
    x11root="/usr/X11"
  elif [ -d /usr/X11R6/lib ]; then
    x11root="/usr/X11R6"
  fi

  if [ -n "$x11root" ]; then

    if   [ -d "$x11root/lib${SYS_BITS}" ]; then
      dir="$x11root/lib${SYS_BITS}"
    elif [ -d "$x11root/lib" ]; then
      dir="$x11root/lib"
    fi

    if FindFiles --silent "libX11*" "$dir"; then
      GL_LIBDIRS="$dir"
    fi

  fi

  for dir in /usr/local/lib /usr/local/lib${SYS_BITS}; do
    if FindFiles --silent "libglut*" $dir; then
      GL_LIBDIRS="$GL_LIBDIRS $dir"
      break
    fi
  done

  if [ -d /usr/local/include/GL ]; then
    GL_INCDIRS="$GL_INCDIRS /usr/local/include"
  fi

  # First try without including /usr/X11/include as GCC may
  # use a fixed header file.

  if TestGL --silent; then
    return
  fi

  if [ -n "$x11root" ] && [ -d "$x11root/include" ]; then
    GL_INCDIRS="$x11root/include"
  fi

  return 0
}


#========================================================================
#   Main script
#========================================================================


SetLinuxGLDefaults
