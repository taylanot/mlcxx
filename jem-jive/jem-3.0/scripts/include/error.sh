
#@skip

if [ -n "$JEM_SCRIPTS_ERROR_SH" ]; then
  return 0
else
  JEM_SCRIPTS_ERROR_SH=1
fi

#@endskip


#========================================================================
#   functions
#========================================================================

#------------------------------------------------------------------------
#   Oops ( %line1 [%line2 ...] )
#------------------------------------------------------------------------


Oops()
{
  local prog="$PROGNAME"
  local i=""

  if [ -z "$prog" ]; then
    prog="$(basename $0)"
  fi

  echo "" 1>&2
  echo "*** Oops, internal error in \`$prog':" 1>&2
  echo "" 1>&2

  for i in "$@" ; do
    echo "    $i" 1>&2
  done

  echo "" 1>&2
  echo "    This error should not have happened. Please send a" 1>&2
  echo "    bug report to DRG (www.dynaflow.com)." 1>&2
  echo "" 1>&2

  exit 1
}


#------------------------------------------------------------------------
#   Error ( %line1 [%line2 ...] )
#------------------------------------------------------------------------


Error()
{
  local i=""

  echo "" 1>&2
  echo "*** Error:" 1>&2
  echo "" 1>&2

  for i in "$@" ; do
    echo "    $i" 1>&2
  done

  echo "" 1>&2

  exit 1
}


#------------------------------------------------------------------------
#   OptionError ( %option )
#------------------------------------------------------------------------


OptionError()
{
  local prog="$PROGNAME"

  if [ -z "$prog" ]; then
    prog="$(basename $0)"
  fi

  echo "$prog: invalid option: $1; try \`$prog --help'." 1>&2

  exit 1
}


#------------------------------------------------------------------------
#   InputError ( %text )
#------------------------------------------------------------------------


InputError()
{
  local prog="$PROGNAME"

  if [ -z "$prog" ]; then
    prog="$(basename $0)"
  fi

  echo "$prog: $1; try \`$prog --help'." 1>&2

  exit 1
}
