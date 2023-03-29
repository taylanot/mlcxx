
#------------------------------------------------------------------------
#   PrintMakefile ()
#------------------------------------------------------------------------


PrintMakefile()
{
  cat<<EOF

JIVEPATH := \$(shell "\$(JEMDIR)/bin/pathto" "\$(JIVEDIR)")

include \$(JIVEPATH)/makefiles/jive.mk
include \$(JIVEPATH)/makefiles/private.mk
include \$(JIVEPATH)/makefiles/pkg.mk

EOF
}

