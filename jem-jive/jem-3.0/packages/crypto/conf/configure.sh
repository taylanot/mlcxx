
#========================================================================
#   Global variables
#========================================================================


USE_CRYPTO=1

CRYPTO_LIBS="crypto"
CRYPTO_LIBDIRS=""
CRYPTO_INCDIRS=""


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

      --crypto-libs=?*)

        CRYPTO_LIBS="${arg#--crypto-libs=}" ;;

      --crypto-libdirs=?*)

        CRYPTO_LIBDIRS="${arg#--crypto-libdirs=}" ;;

      --crypto-incdirs=?*)

        CRYPTO_INCDIRS="${arg#--crypto-incdirs=}" ;;

      --with-crypto)

        USE_CRYPTO=1 ;;

      --without-crypto)

        USE_CRYPTO="" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestCrypto ()
#------------------------------------------------------------------------


TestCrypto()
{
  local f="$(FindSysconfFile test-crypto.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$CRYPTO_LIBS"
  MY_LIBDIRS="$CRYPTO_LIBDIRS"
  MY_INCDIRS="$CRYPTO_INCDIRS"

  Log    "Testing crypto setup ..."

  printf "Testing if crypto works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    return 0
  else
    echo "no."
    return 1
  fi
}


#------------------------------------------------------------------------
#   PrintConfig ()
#------------------------------------------------------------------------


PrintConfig()
{
  Log "Saving crypto configuration."

  echo "# Generated on $(date)."
  echo "# CHANGES TO THIS FILE WILL BE LOST."
  echo ""

  echo "CRYPTO_LIBS=$(quote "$CRYPTO_LIBS")"
  echo "CRYPTO_LIBDIRS=$(quote "$CRYPTO_LIBDIRS")"
  echo "CRYPTO_INCDIRS=$(quote "$CRYPTO_INCDIRS")"
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_CRYPTO" ] && TestCrypto ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-crypto"
    echo "  --crypto-libs=$(quote "$CRYPTO_LIBS")"
    echo "  --crypto-libdirs=$(quote "$CRYPTO_LIBDIRS")"
    echo "  --crypto-incdirs=$(quote "$CRYPTO_INCDIRS")"
    echo ""

  fi

  WriteFile PrintConfig conf/config.sh

  Log "Enabled crypto support."

  MY_LIBS="$CRYPTO_LIBS"
  MY_LIBDIRS="$CRYPTO_LIBDIRS"
  MY_INCDIRS="$CRYPTO_INCDIRS"

else

  LogEcho "Disabled the crypto package."

  exit 0

fi
