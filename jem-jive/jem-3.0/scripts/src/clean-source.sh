
#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] DIRS

Recursively deletes all C++ and Doxygen source files in the
directories DIRS.

Options:

  -h,--help    : print this help.
  -v,--verbose : show what is going on.
  -f,--force   : do not ask for confirmation.

EOF
}


#------------------------------------------------------------------------
#   Run ( [%arg1 %arg2 ...] )
#------------------------------------------------------------------------


Run()
{
  local verbose=""
  local force=""
  local ans=""
  local f=""

  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

     -v|--verbose)

        verbose=1 ;;

     -f|--force)

        force=1 ;;

     --)

        shift
        break ;;

     -*)

        echo "$0: invalid option: $1; try \`$0 --help'." 1>&2
        exit 1 ;;

     *)

        break ;;

    esac

    shift

  done

  if [ -z "$force" ]; then

    printf "Are you sure you want to delete all source files (y|n)? "
    read ans

    case "$ans" in
      y|yes|Y|YES) :      ;;
      *)           exit 0 ;;
    esac

  fi

  while [ $# -gt 0 ]; do

    if [ ! -d "$1" ]; then
      echo "$0: no such directory: $1." 1>&2
      exit 1
    fi

    find "$1" \( -name examples -prune \) -o \
              \( -name "*.cpp" -o -name "*.dox" \) -print | \
    while read f; do
      if [ -n "$verbose" ]; then
        echo "Deleting \`$f' ..."
      fi
      rm -f -- "$f"
    done

    shift

  done
}


#========================================================================
#   Main script
#========================================================================

Run "$@"
