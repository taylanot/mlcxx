
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_APP_CONTROLMODULE_H
#define JIVE_APP_CONTROLMODULE_H

#include <jive/app/Module.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class ControlModule
//-----------------------------------------------------------------------


class ControlModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( ControlModule, Module );

  static const char*        TYPE_NAME;

  static const int          FG_MODE;


  explicit                  ControlModule

    ( const String&           name  = "control",
      const Ref<Module>&      child = nullptr,
      int                     mode  = 0 );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat )            override;

  virtual Status            run

    ( const Properties&       globdat )            override;

  virtual void              shutdown

    ( const Properties&       globdat )            override;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat )            override;

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )      const override;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat )            override;

  void                      setMode

    ( int                     mode );

  inline int                getMode       () const;
  String                    getPrompt     () const;

  void                      setPrompt

    ( const String&           prompt );

  void                      runWhile

    ( const String&           expr );

  void                      printHelp

    ( PrintWriter&            out )          const;

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();


 protected:

  virtual                  ~ControlModule ();


 private:

  void                      getCommands_

    ( const Properties&       cmds,
      const Properties&       globdat );

  bool                      tryExeCmd_

    ( const String&           cmd,
      const String&           arg,
      const Properties&       globdat );

  void                      exeCommand_

    ( const String&           cmd,
      const String&           arg,
      const Properties&       globdat );

  void                      exeSetCmd_

    ( const String&           arg,
      const Properties&       globdat );

  void                      exeEvalCmd_

    ( const String&           arg,
      const Properties&       globdat );

  void                      exeShowCmd_

    ( const String&           arg,
      const Properties&       globdat );

  void                      exeHelpCmd_

    ( const String&           arg,
      const Properties&       globdat );

  static void               splitCommand_

    ( String&                 cmd,
      String&                 arg );


 private:

  class                     RunData_;
  class                     RootData_;
  class                     ProcData_;
  class                     Completer_;

  friend class              RootData_;
  friend class              ProcData_;


  Ref<Module>               child_;

  String                    prompt_;
  String                    cmdFile_;

  int                       mode_;
  double                    pauseTime_;

  Ref<RunData_>             rundat_;
  Ref<Function>             runCond_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getMode
//-----------------------------------------------------------------------


inline int ControlModule::getMode () const
{
  return mode_;
}


JIVE_END_PACKAGE( app )

#endif
