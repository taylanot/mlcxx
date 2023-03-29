
#########################################################################
# Pattern rules
#########################################################################


$(OBJBASE)%$(OBJ_SUFFIX):	%.cpp
				$(Compile)

$(OBJBASE_OPT)%$(OBJ_SUFFIX): 	%.cpp
				$(Compile)

$(OBJBASE_DBG)%$(OBJ_SUFFIX):	%.cpp
				$(Compile)

$(OBJBASE_OPT)%$(OBJ_SUFFIX):	CMODE = $(CMODE_OPT)

$(OBJBASE_DBG)%$(OBJ_SUFFIX): 	CMODE = $(CMODE_DBG)


#########################################################################
# Object dir rules
#########################################################################


ifneq ($(objects),)


ifdef OBJDIR

tagfile			:= $(OBJDIR)/MAKE.TMP
tagfile-opt		:= $(OBJDIR_OPT)/MAKE.TMP
tagfile-dbg		:= $(OBJDIR_DBG)/MAKE.TMP


$(objects):		$(tagfile)

$(objects-opt):		$(tagfile-opt)

$(objects-dbg):		$(tagfile-dbg)


$(tagfile):
			@test -d $(OBJDIR) ||\
			  $(MKDIR) $(OBJDIR)
			@echo "Make was here" > $(tagfile)

$(tagfile-opt):
			@test -d $(OBJDIR_OPT) ||\
			  $(MKDIR) $(OBJDIR_OPT)
			@echo "Make was here" > $(tagfile-opt)

$(tagfile-dbg):
			@test -d $(OBJDIR_DBG) ||\
			  $(MKDIR) $(OBJDIR_DBG)
			@echo "Make was here" > $(tagfile-dbg)

endif


#########################################################################
# Definitions
#########################################################################


define CleanObjFiles
  for i in $(TEMP_DIRS); do \
    if [ -d $$i ]; then \
      $(RM_R) $$i ; \
    fi ; \
  done
endef


endif
