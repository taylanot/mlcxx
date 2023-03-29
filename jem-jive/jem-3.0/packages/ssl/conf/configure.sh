
#========================================================================
#   Global variables
#========================================================================


USE_SSL=1

SSL_LIBS="ssl"
SSL_LIBDIRS=""
SSL_INCDIRS=""


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

      --ssl-libs=?*)

        SSL_LIBS="${arg#--ssl-libs=}" ;;

      --ssl-libdirs=?*)

        SSL_LIBDIRS="${arg#--ssl-libdirs=}" ;;

      --ssl-incdirs=?*)

        SSL_INCDIRS="${arg#--ssl-incdirs=}" ;;

      --with-ssl)

        USE_SSL=1 ;;

      --without-ssl)

        USE_SSL="" ;;

    esac

  done
}


#------------------------------------------------------------------------
#   TestSSL ()
#------------------------------------------------------------------------


TestSSL()
{
  local f="$(FindSysconfFile test-ssl.cpp)"

  if [ -z "$f" ]; then
    return 1
  fi

  MY_LIBS="$SSL_LIBS $CRYPTO_LIBS"
  MY_LIBDIRS="$SSL_LIBDIRS $CRYPTO_LIBDIRS"
  MY_INCDIRS="$SSL_INCDIRS $CRYPTO_INCDIRS"

  Log    "Testing SSL setup ..."

  printf "Testing if SSL works ... "

  if TestSyslibs "$f" ; then
    echo "yes."
    return 0
  else
    echo "no."
    return 1
  fi
}


#========================================================================
#   Main script
#========================================================================


SetDefaults "$@"
ParseArgs "$@"

if [ -n "$USE_SSL" ]; then

  if [ ! -e "$PKG_DIR/../crypto/conf/config.sh" ]; then
    Log "The crypto package has not been configured yet."
    USE_SSL=""
  else
    Log "Loading the crypto package configuration."
    . "$PKG_DIR/../crypto/conf/config.sh"
  fi

fi

if [ -n "$USE_SSL" ] && TestSSL ; then

  if [ -n "$VERBOSE" ]; then

    echo ""
    echo "  --with-ssl"
    echo "  --ssl-libs=$(quote "$SSL_LIBS")"
    echo "  --ssl-libdirs=$(quote "$SSL_LIBDIRS")"
    echo "  --ssl-incdirs=$(quote "$SSL_INCDIRS")"
    echo ""

  fi

  Log "Enabled SSL support."

  MY_LIBS="$SSL_LIBS"
  MY_LIBDIRS="$SSL_LIBDIRS"
  MY_INCDIRS="$SSL_INCDIRS"

else

  LogEcho "Disabled the ssl package."

  exit 0

fi
