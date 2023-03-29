
#@skip

if [ -n "$JEM_SCRIPTS_UTILS_SH" ]; then
  return 0
else
  JEM_SCRIPTS_UTILS_SH=1
fi

#@endskip


#========================================================================
#   Global variables
#========================================================================


DIR_STACK_DEPTH=0


#========================================================================
#   functions
#========================================================================

#------------------------------------------------------------------------
#   Die ( [%exitCode] )
#------------------------------------------------------------------------


Die()
{
  if [ -z "$1" ]; then
    exit 1
  else
    exit "$1"
  fi
}


#------------------------------------------------------------------------
#   OpenLogfile ( %fname )
#------------------------------------------------------------------------


OpenLogfile()
{
  if [ -n "$JEM_LOGFILE" ]; then
    return
  fi

  export JEM_LOGFILE="$1"

  if [ -n "$JEM_LOGFILE" ]; then
    echo "Logfile created by \`$0' on $(date)." >  "$JEM_LOGFILE"
    echo "Current directory: $(pwd)."           >> "$JEM_LOGFILE"
  fi
}


#------------------------------------------------------------------------
#   CloseLogfile ()
#------------------------------------------------------------------------


CloseLogfile()
{
  if [ -n "$JEM_LOGFILE" ]; then
    echo "Logfile closed on $(date)." >> "$JEM_LOGFILE"
    JEM_LOGFILE=""
  fi
}


#------------------------------------------------------------------------
#   Log ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Log()
{
  if [ -n "$JEM_LOGFILE" ]; then
    echo "$@" >> "$JEM_LOGFILE"
  fi
}


#------------------------------------------------------------------------
#   LogEcho ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


LogEcho()
{
  if [ -n "$JEM_LOGFILE" ]; then
    echo "$@" >> "$JEM_LOGFILE"
  fi

  echo "$@"
}


#------------------------------------------------------------------------
#   PushDir ( %dir )
#------------------------------------------------------------------------


PushDir()
{
  eval DIR_STACK_ENTRY_$DIR_STACK_DEPTH='"$(pwd)"'

  cd "$1" || return 1

  DIR_STACK_DEPTH=$(( $DIR_STACK_DEPTH + 1 ))

  if [ -n "$JEM_LOGFILE" ]; then
    echo "Entered directory \`$(pwd)'." >> "$JEM_LOGFILE"
  fi

  return 0
}


#------------------------------------------------------------------------
#   PopDir ()
#------------------------------------------------------------------------


PopDir()
{
  if [ $DIR_STACK_DEPTH -le 0 ]; then
    echo "$0: empty directory stack." 1>&2
    return 1
  fi

  DIR_STACK_DEPTH=$(( $DIR_STACK_DEPTH - 1 ))

  eval cd \"\$DIR_STACK_ENTRY_$DIR_STACK_DEPTH\" || return 1

  if [ -n "$JEM_LOGFILE" ]; then
    echo "Returned to directory \`$(pwd)'." >> "$JEM_LOGFILE"
  fi

  return 0
}


#------------------------------------------------------------------------
#   ChangeDir ( %dir )
#------------------------------------------------------------------------


ChangeDir()
{
  cd "$1" || return 1

  if [ -n "$JEM_LOGFILE" ]; then
    echo "Changed to directory \`$(pwd)'." >> "$JEM_LOGFILE"
  fi

  return 0
}


#------------------------------------------------------------------------
#   MakeDir ( %dir )
#------------------------------------------------------------------------


MakeDir()
{
  mkdir -p -- "$1" || return 1

  if [ -n "$JEM_LOGFILE" ]; then
    echo "Created directory \`$1'." >> "$JEM_LOGFILE"
  fi

  return 0
}


#------------------------------------------------------------------------
#   ToUpper ( %word )
#------------------------------------------------------------------------


ToUpper()
{
  echo "$1" | tr "a-z" "A-Z"
}


#------------------------------------------------------------------------
#   ToLower ( %word )
#------------------------------------------------------------------------


ToLower()
{
  echo "$1" | tr "A-Z" "a-z"
}


#------------------------------------------------------------------------
#   FindWord ( [-s|--silent] [--] %pattern [%word1 %word2 ...] )
#------------------------------------------------------------------------


FindWord()
{
  local silent=""
  local pattern=""
  local word=""

  while [ $# -gt 0 ]; do

    case "$1" in

      -s|--silent)

        silent=1 ;;

      --)

        shift
        break ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    return 1
  fi

  pattern="$1"
  shift

  for word in "$@"; do
    case "$word" in
      $pattern)
        if [ -z "$silent" ]; then
          echo "$word"
        fi
        return 0 ;;
    esac
  done

  return 1
}


#------------------------------------------------------------------------
#   FindExe ( [-s|--silent] [--] %fname )
#------------------------------------------------------------------------


FindExe()
{
  local silent=""
  local fname=""
  local fpath=""
  local dir=""

  while [ $# -gt 0 ]; do

    case "$1" in

      -s|--silent)

        silent=1 ;;

      --)

        shift
        break ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    return 1
  fi

  fname="$1"
  shift

  # This seemingly complicated procedure will preserve all
  # special characters in $PATH.

  fpath=\'"$(echo "$PATH" | sed -e "s/'/'\\\\''/g" \
                          | sed -e "s/:/' '/g")"\'

  eval "set -- $fpath"
  fpath=""

  for dir in "$@"; do
    if [ -x "$dir/$fname" ]; then
      if [ -z "$silent" ]; then
        echo "$dir/$fname"
      fi
      return 0
    fi
  done

  return 1
}


#------------------------------------------------------------------------
#   FindFile ( [-s|--silent] [--] %fname [%dir1 %dir2 ...] )
#------------------------------------------------------------------------


FindFile()
{
  local silent=""
  local fname=""
  local dir=""

  while [ $# -gt 0 ]; do

    case "$1" in

      -s|--silent)

        silent=1 ;;

      --)

        shift
        break ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    return 1
  fi

  fname="$1"
  shift

  for dir in "$@" ; do
    if [ -e "$dir/$fname" ]; then
      if [ -z "$silent" ]; then
        echo "$dir/$fname"
      fi
      return 0
    fi
  done

  return 1
}


#------------------------------------------------------------------------
#   FindFiles ( [-s|--silent] [--] %pattern [%dir1 %dir2 ...] )
#------------------------------------------------------------------------


FindFiles()
{
  local pattern=""
  local silent=""
  local result=1
  local dir=""
  local f=""

  while [ $# -gt 0 ]; do

    case "$1" in

      -s|--silent)

        silent=1 ;;

      --)

        shift
        break ;;

      *)

        break ;;

    esac

    shift

  done

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    return 1
  fi

  pattern="$1"
  shift

  if [ $# -lt 1 ] || [ -z "$1" ]; then
    set -- "."
  fi

  for dir in "$@"; do
    for f in "$dir"/$pattern ; do
      if [ -e "$f" ]; then
        if [ -n "$silent" ]; then
          return 0
        fi
        result=0
        echo "$f"
      fi
    done
  done

  return $result
}


#------------------------------------------------------------------------
#   ReadConfigData ()
#------------------------------------------------------------------------


ReadConfigData()
{
  if [ ! -e "$JEMDIR/scripts/include/config.sh" ] ; then
    echo "$0: it seems that Jem has not yet been configured." 1>&2
    echo "Please run \`\$JEMDIR/configure' first."            1>&2
    return 1
  fi

  . "$JEMDIR/scripts/include/config.sh"
}


#------------------------------------------------------------------------
#   GetPackageName ()
#------------------------------------------------------------------------


GetPackageName()
{
  local name="$(pwd)"

  name="/${name#/}"

  case "$name" in

    */packages/*)

      name="${name##*/packages/}"
      name="${name%%/*}" ;;

    *)

      name="" ;;

  esac

  if [ -z "$name" ]; then
    return 1
  fi

  echo "$name"
}
