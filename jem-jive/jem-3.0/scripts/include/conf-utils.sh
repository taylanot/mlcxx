
#@include "${JEMDIR:?}/scripts/include/error.sh"
#@include "${JEMDIR:?}/scripts/include/utils.sh"

#@skip

if [ -n "$JEM_SCRIPTS_CONF_UTILS_SH" ]; then
  return 0
else
  JEM_SCRIPTS_CONF_UTILS_SH=1
fi

. "${JEMDIR:?}/scripts/include/error.sh"
. "${JEMDIR:?}/scripts/include/utils.sh"

#@endskip


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   CheckArgs ()
#------------------------------------------------------------------------


CheckArgs()
{
  local args="$(PrintArgs)"
  local i=""
  local j=""

  for i in "$@" ; do

    for j in $args ; do
      if [ "xx${i%%=*}" = "xx${j%%=*}" ]; then
        i=""
      fi
    done

    if [ -n "$i" ]; then
      OptionError "$i"
    fi

  done
}


#------------------------------------------------------------------------
#   WriteFile ( %cmd %output )
#------------------------------------------------------------------------


WriteFile()
{
  LogEcho "Writing \`$2' ..."

  eval "$1" '> "$2"'
}


#------------------------------------------------------------------------
#   ListConfigDirs ( [--no-cxx] [%baseDir] )
#------------------------------------------------------------------------


ListConfigDirs()
{
  local cxx="$CXX_NAME"
  local base=""
  local dirs=""
  local d=""

  if [ "xx$1" = "xx--no-cxx" ]; then
    cxx=""
    shift
  fi

  base="${1:-.}"
  d="$base/conf"

  if [ -d "$d" ]; then
    dirs="$d"
  fi

  if [ -n "$SYS_IS_POSIX" ]; then

    d="$base/conf/posix"

    if [ -d "$d" ]; then
      dirs="$dirs $d"
    fi

  fi

  if [ -n "$SYS_NAME" ]; then

    d="$base/conf/$SYS_NAME"

    if [ -d "$d" ]; then
      dirs="$dirs $d"
    fi

  fi

  if [ -n "$cxx" ]; then

    d="$base/conf/${SYS_NAME}-${cxx}"

    if [ -d "$d" ]; then
      dirs="$dirs $d"
    fi

  fi

  echo "$dirs"
}


#------------------------------------------------------------------------
#   ListSysconfDirs
#------------------------------------------------------------------------


alias ListSysconfDirs='ListConfigDirs --no-cxx'


#------------------------------------------------------------------------
#   FindConfigFile ( %fname [%baseDir] )
#------------------------------------------------------------------------


FindConfigFile()
{
  local base="${2:-.}"
  local f=""
  local p=""

  p="$base/conf/$1"

  if [ -e "$p" ]; then
    f="$p"
  fi

  if [ -n "$SYS_IS_POSIX" ]; then

    p="$base/conf/posix/$1"

    if [ -e "$p" ]; then
      f="$p"
    fi
  fi

  if [ -n "$SYS_NAME" ]; then

    p="$base/conf/$SYS_NAME/$1"

    if [ -e "$p" ]; then
      f="$p"
    fi
  fi

  if [ -n "$CXX_NAME" ]; then

    p="$base/conf/${SYS_NAME}-${CXX_NAME}/$1"

    if [ -e "$p" ]; then
      f="$p"
    fi
  fi

  echo "$f"

  if [ -n "$f" ]; then
    return 0
  else
    return 1
  fi
}


#------------------------------------------------------------------------
#   FindSysconfFile ( %fname [%baseDir] )
#------------------------------------------------------------------------


FindSysconfFile()
{
  local base="${2:-.}"
  local f=""
  local p=""

  p="$base/conf/$1"

  if [ -e "$p" ]; then
    f="$p"
  fi

  if [ -n "$SYS_IS_POSIX" ]; then

    p="$base/conf/posix/$1"

    if [ -e "$p" ]; then
      f="$p"
    fi
  fi

  if [ -n "$SYS_NAME" ]; then

    p="$base/conf/$SYS_NAME/$1"

    if [ -e "$p" ]; then
      f="$p"
    fi
  fi

  echo "$f"

  if [ -n "$f" ]; then
    return 0
  else
    return 1
  fi
}


#------------------------------------------------------------------------
#   ListPackageDeps ( %pkg )
#------------------------------------------------------------------------


ListPackageDeps()
{
  case "$1" in

    base|io|util|std)

      echo "$1"
      return ;;

  esac

  if [ -e "$JEMDIR/packages/$1/SPECS.sh" ]; then

    (
      . "$JEMDIR/packages/$1/SPECS.sh"

      for p in $PKG_DEPS; do
        ListPackageDeps "$p"
      done
    )

  fi

  echo "$1"
}


#------------------------------------------------------------------------
#   UnconfigureJem ()
#------------------------------------------------------------------------


UnconfigureJem()
{
  local dir=""
  local pkg=""
  local f=""

  if [ -n "$VERBOSE" ]; then
    echo "Removing files ..."
  fi

  rm -f -- "$JEMDIR"/makefiles/config.mk \
           "$JEMDIR"/scripts/include/config.sh

  if [ -d "$JEMDIR"/scripts/bin ]; then
    rm -f -- "$JEMDIR"/scripts/bin/*
  fi

  if [ -d "$JEMDIR"/bin ]; then
    rm -f -- "$JEMDIR"/bin/*
  fi

  if [ -d "$JEMDIR"/makefiles/packages ]; then
    rm -f -- "$JEMDIR"/makefiles/packages/*
  fi

  if [ -n "$VERBOSE" ]; then
    echo "Removing libraries ..."
  fi

  for dir in lib lib-opt lib-dbg ; do
    if [ -d "$JEMDIR/$dir" ]; then
      rm -f -- "$JEMDIR/$dir"/*
    fi
  done

  if [ -d "$JEMDIR"/packages ]; then

    if [ -n "$VERBOSE" ]; then
      printf "Unconfiguring installed packages:"
    fi

    for dir in "$JEMDIR"/packages/* ; do

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
  local dir="$JEMDIR/packages/$pkg"
  local f=""

  if [ ! -d "$dir" ]; then
    return 1
  fi

  PushDir "$dir" || return 1

  PKG_NAME="$pkg"
  PKG_DIR="$dir"
  PKG_INC_DIR="$JEMDIR/include/jem/$pkg"

  if [ -e Makefile ]; then
    rm -f NoMakefile
    mv    Makefile NoMakefile
  fi

  rm -f -- Makedefs.mk \
           "$JEMDIR/makefiles/packages/${pkg}.mk"

  if f="$(FindSysconfFile unconfigure.sh)" ; then
    . "$f"
  fi

  PopDir || return 1
}


#------------------------------------------------------------------------
#   TestSyslibs ( %srcFile )
#------------------------------------------------------------------------


TestSyslibs()
{
  local src="$1"
  local exe="${1%*.cpp}$EXE_SUFFIX"
  local obj="${1%*.cpp}$OBJ_SUFFIX"

  "$JEMDIR/bin/CC" -o "$exe" \
                   -I "$MY_INCDIRS $SYS_INCDIRS" \
                   -L "$MY_LIBDIRS $SYS_LIBDIRS" \
                   -l "$MY_LIBS $SYS_LIBS" \
                   -- "$src" >> "$JEM_LOGFILE" 2>&1

  rm -f -- "$obj"

  if [ -x "$exe" ]; then
    rm -f -- "$exe"
    return 0
  else
    return 1
  fi
}
