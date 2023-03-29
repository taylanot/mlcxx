
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
  echo "Enables the $TOOLKIT package $1."
}


#------------------------------------------------------------------------
#   Run ()
#------------------------------------------------------------------------


Run()
{
  ChangeDir "$PKG_DIR" || Die

  rm -f DISABLED

  if [ -e NoMakefile ]; then
    rm -f Makefile
    mv    NoMakefile Makefile
  fi

  if [ -e "$PKG_DIR/Makedefs.mk" ]; then
    cp -f -- "$PKG_DIR/Makedefs.mk" \
             "$TOPDIR/makefiles/packages/${PKG_NAME}.mk"
  fi

  if [ -n "$VERBOSE" ]; then

    if [ -e "$PKG_DIR/Makedefs.mk" ]; then
      echo "The package \`$PKG_NAME' has been enabled."
      echo "Run \`disable-package $PKG_NAME' to disable it again."
    else
      echo "The package \`$PKG_NAME' has not yet been configured;"
      echo "run \`configure-package $PKG_NAME' first."
    fi

  fi
}


#========================================================================
#   Main script
#========================================================================


. "${JEMDIR:?}/scripts/include/pkgtool.sh"
