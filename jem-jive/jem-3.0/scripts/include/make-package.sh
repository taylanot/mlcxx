
#------------------------------------------------------------------------
#   PrintMakefile ()
#------------------------------------------------------------------------


PrintMakefile()
{
  cat<<EOF

JEMPATH := \$(shell "\$(JEMDIR)/bin/pathto" "\$(JEMDIR)")

include \$(JEMPATH)/makefiles/jem.mk
include \$(JEMPATH)/makefiles/private.mk
include \$(JEMPATH)/makefiles/pkg.mk

EOF
}
