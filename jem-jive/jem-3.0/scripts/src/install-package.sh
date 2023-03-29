
#@set TOPDIR="${TOPDIR:?}"
#@set TOOLKIT="${TOOLKIT:?}"

alias pathto="$JEMDIR/bin/pathto"


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintInfo ( %pkg )
#------------------------------------------------------------------------


PrintInfo()
{
  echo "Installs the $TOOLKIT package $1."
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  if [ -e "$PKG_DIR/Makedefs.mk" ]; then
    cp -f -- "$PKG_DIR/Makedefs.mk" \
             "$TOPDIR/makefiles/packages/${PKG_NAME}.mk"
  fi

  if [ -n "$VERBOSE" ]; then
    echo "The package \`$PKG_NAME' has been successfully installed."
  fi
}


#========================================================================
#   Main script
#========================================================================


. "${JEMDIR:?}/scripts/include/pkgtool.sh"
