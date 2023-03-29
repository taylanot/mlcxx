
#@include "${JEMDIR:?}/scripts/include/utils.sh"

#@skip

if [ -n "$JIVE_SCRIPTS_CONF_UTILS_SH" ]; then
  return 0
else
  JIVE_SCRIPTS_CONF_UTILS_SH=1
fi

. "${JEMDIR:?}/scripts/include/utils.sh"

#@endskip


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   WriteFile ( %cmd %output )
#------------------------------------------------------------------------


WriteFile()
{
  eval "$1" '> "$2"'

  Log "Wrote \`$2'."
}


#------------------------------------------------------------------------
#   ListPackageDeps ( %pkg )
#------------------------------------------------------------------------


ListPackageDeps()
{
  local p="$1"

  case "$p" in

    jem.*)

      return ;;

    jive.*)

      p="${p#jive.}" ;;

  esac

  if [ -e "$JIVEDIR/packages/$p/SPECS.sh" ]; then

    (
      . "$JIVEDIR/packages/$p/SPECS.sh"

      for i in $PKG_DEPS; do
        ListPackageDeps "$i"
      done
    )

  fi

  echo "$p"
}


#------------------------------------------------------------------------
#   UnconfigureJive ()
#------------------------------------------------------------------------


UnconfigureJive()
{
  local dir=""
  local pkg=""
  local f=""

  if [ -n "$VERBOSE" ]; then
    echo "Removing files ..."
  fi

  rm -f -- "$JIVEDIR"/scripts/include/config.sh

  if [ -d "$JIVEDIR"/scripts/bin ]; then
    rm -f -- "$JIVEDIR"/scripts/bin/*
  fi

  if [ -d "$JIVEDIR"/bin ]; then
    rm -f -- "$JIVEDIR"/bin/*
  fi

  if [ -d "$JIVEDIR"/makefiles/packages ]; then
    rm -f -- "$JIVEDIR"/makefiles/packages/*
  fi

  if [ -n "$VERBOSE" ]; then
    echo "Removing libraries ..."
  fi

  for dir in lib lib-opt lib-dbg ; do
    if [ -d "$JIVEDIR/$dir" ]; then
      rm -f -- "$JIVEDIR/$dir"/*
    fi
  done

  if [ -d "$JIVEDIR"/packages ]; then

    if [ -n "$VERBOSE" ]; then
      printf "Unconfiguring installed packages:"
    fi

    for dir in "$JIVEDIR"/packages/* ; do

      if [ -d "$dir" ]; then

        pkg="$(basename "$dir")"

        UnconfigurePackage "$pkg"

        if [ -n "$VERBOSE" ]; then
          printf " $pkg"
        fi

      fi

    done

  fi

  if [ -n "$VERBOSE" ]; then
    echo ""
  fi
}


#------------------------------------------------------------------------
#   UnconfigurePackage ( %pkg )
#------------------------------------------------------------------------


UnconfigurePackage()
{
  local pkg="$1"
  local dir="$JIVEDIR/packages/$pkg"
  local f=""

  if [ ! -d "$dir" ]; then
    return 1
  fi

  PushDir "$dir" || return 1

  if [ -e Makefile ]; then
    rm -f NoMakefile
    mv    Makefile NoMakefile
  fi

  rm -f -- Makedefs.mk \
           "$JIVEDIR/makefiles/packages/${pkg}.mk"

  PopDir || return 1
}
