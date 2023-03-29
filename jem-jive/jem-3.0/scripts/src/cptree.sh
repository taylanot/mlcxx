
#========================================================================
# Variables
#========================================================================


VERBOSE=""
EXCL_LIST=""
SRC=""
DEST=""


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintHelp ()
#------------------------------------------------------------------------


PrintHelp()
{
  cat<<EOF

Usage: $(basename "$0") [OPTIONS] [--] SRC DEST

Recursively copies the file or directory SRC to DEST/SRC.

Options:

  -h, --help     : show this help.

  -v, --verbose  : show what is going on.

  -x PATTERN     : exclude files and directories matching PATTERN.

EOF
}


#------------------------------------------------------------------------
#   ParseArgs()
#------------------------------------------------------------------------


ParseArgs()
{
  while [ $# -gt 0 ]; do

    case "$1" in

      -h|--help)

        PrintHelp
        exit 0 ;;

      -v|--vervbose)

        VERBOSE=1 ;;

      -x)

        shift
        EXCL_LIST="$EXCL_LIST --exclude=$(quote "$1")" ;;

      -r?*)

        EXCL_LIST="$EXCL_LIST --exclude=$(quote "${1#-r}")" ;;

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

  if [ $# -lt 1 ]; then
    echo "$0: missing source directory; try \`$0 --help'." 1>&2
    exit 1
  fi

  if [ $# -lt 2 ]; then
    echo "$0: missing destination directory; try \`$0 --help'." 1>&2
    exit 1
  fi

  if [ $# -gt 2 ]; then
    echo "$0: too many arguments; try \`$0 --help'." 1>&2
    exit 1
  fi

  SRC="$1"
  DEST="$2"
}


#------------------------------------------------------------------------
#   Run()
#------------------------------------------------------------------------


Run()
{
  local xflags=""
  local cmd=""

  if [ -z "$VERBOSE" ]; then
    xflags="xf -"
  else
    xflags="xvf -"
  fi

  if [ ! -d "$DEST" ]; then
    "$0: the destination \`$DEST' is not a directory."
    exit 1
  fi

  cmd="tar cf - $EXCL_LIST -C $(quote "$(dirname "$SRC")")"
  cmd="$cmd $(quote "$(basename "$SRC")")"
  cmd="$cmd | tar $xflags -C $(quote "$DEST")"

  eval "$cmd"

  return $?
}


#========================================================================
#   Main script
#========================================================================


ParseArgs  "$@"
Run
