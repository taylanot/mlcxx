
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

        LIBS="$LIBS -l$(quote "$i")" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   EvalLibdirs ()
#------------------------------------------------------------------------


EvalLibdirs()
{
  local i=""

  eval "set -- $LIBDIRS"

  LIBDIRS=""

  for i in "$@" ; do

    case "$i" in

      -NODIR:?*)

        LIBDIRS="$LIBDIRS $(quote "${i#-NODIR:}")" ;;

      *)

        LIBDIRS="$LIBDIRS -L$(quote "$i")" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   EvalIncdirs ()
#------------------------------------------------------------------------


EvalIncdirs()
{
  local i=""

  eval "set -- $INCDIRS"

  INCDIRS=""

  for i in "$@" ; do

    case "$i" in

      -NODIR:?*)

        INCDIRS="$INCDIRS $(quote "${i#-NODIR:}")" ;;

      *)

        INCDIRS="$INCDIRS -I$(quote "$i")" ;;

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

  cmd="$CXX $FLAGS $INCDIRS $LIBDIRS"

  if [ -n "$COMPILE_ONLY" ]; then
    cmd="$cmd -c"
  fi

  if [ -n "$OUTPUT" ]; then
    cmd="$cmd -o $OUTPUT"
  fi

  cmd="$cmd $INPUT $LIBS"

  eval echo "$cmd"

  if [ -z "$NO_EXEC" ]; then
    eval "$cmd"
  fi
}
