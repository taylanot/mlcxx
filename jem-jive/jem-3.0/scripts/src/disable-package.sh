
#@set TOPDIR="${TOPDIR:?}"
#@set TOOLKIT="${TOOLKIT:?}"


#========================================================================
#   Functions
#========================================================================

#------------------------------------------------------------------------
#   PrintInfo ( %pkg )
#------------------------------------------------------------------------


PrintInfo()
{
  echo "Disables the $TOOLKIT package $1."
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  ChangeDir "$PKG_DIR" || Die

  echo "This package has been disabled." >  DISABLED
  echo "Run \`$TOPDIR/bin/enable-package"\
       "$PKG_NAME' to enable it."        >> DISABLED

  if [ -e Makefile ]; then
    rm -f NoMakefile
    mv    Makefile NoMakefile
  fi

  rm -f -- "$TOPDIR/makefiles/packages/${PKG_NAME}.mk"

  if [ -n "$VERBOSE" ]; then
    echo "The package \`$PKG_NAME' has been disabled."
    echo "Run \`enable-package $PKG_NAME' to enable it again."
  fi
}


#========================================================================
#   Main script
#========================================================================


. "${JEMDIR:?}/scripts/include/pkgtool.sh"
