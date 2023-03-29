
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   EvalLibs ()
#------------------------------------------------------------------------


EvalLibs()
{
  local i=""

  eval "set -- $LIBS"

  LIBS=""

  for i in "$@"; do

    case "$i" in

      -NOLIB:?*)

        LIBS="$LIBS $(quote "${i#-NOLIB:}")" ;;

      *)

        LIBS="$LIBS $(quote "$i").lib" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   EvalLibdirs ()
#------------------------------------------------------------------------


EvalLibdirs()
{
  local i=""
  local p=""

  eval "set -- $LIBDIRS"

  LIBDIRS=""

  for i in "$@" ; do
    if [ -n "$i" ]; then
      p="$(echo "$i" | sed 's/^\/\([a-zA-Z]\)\/\(.*\)/\1:\/\2/')"
      LIBDIRS="${LIBDIRS}\\;$(quote "$p")"
    fi
  done

  if [ -n "$LIB" ]; then
    LIBDIRS="\$LIB\\;$LIBDIRS"
  fi
}


#------------------------------------------------------------------------
#   EvalIncdirs ()
#------------------------------------------------------------------------


EvalIncdirs()
{
  local i=""
  local p=""

  eval "set -- $INCDIRS"

  INCDIRS=""

  for i in "$@" ; do

    case "$i" in

      -NODIR:?*)

        INCDIRS="$INCDIRS $(quote "${i#-NODIR:}")" ;;

      *)

        p="$(echo "$i" | sed 's/^\/\([a-zA-Z]\)\/\(.*\)/\1:\/\2/')"
        INCDIRS="$INCDIRS -I $(quote "$p")" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   Compile ()
#------------------------------------------------------------------------


Compile()
{
  local cmd=""

  if [ -n "$LIBS" ]; then
    EvalLibs
  fi

  if [ -n "$LIBDIRS" ]; then
    EvalLibdirs
  fi

  if [ -n "$INCDIRS" ]; then
    EvalIncdirs
  fi

  cmd="LIB=$LIBDIRS $CXX $FLAGS $INCDIRS"

  if [ -n "$COMPILE_ONLY" ]; then
    cmd="$cmd -c"
  fi

  if [ -n "$OUTPUT" ]; then
    if [ -n "$COMPILE_ONLY" ]; then
      cmd="$cmd -Fo$OUTPUT"
    else
      cmd="$cmd -Fe$OUTPUT"
    fi
  fi

  cmd="$cmd $INPUT $LIBS"

  eval echo "$cmd"

  if [ -z "$NO_EXEC" ]; then
    eval "$cmd"
  fi
}
