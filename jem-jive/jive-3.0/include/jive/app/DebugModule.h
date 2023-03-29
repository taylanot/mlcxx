
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

#ifndef JIVE_APP_DEBUGMODULE_H
#define JIVE_APP_DEBUGMODULE_H

#include <jem/util/Timer.h>
#include <jive/app/import.h>
#include <jive/app/Module.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   class DebugModule
//-----------------------------------------------------------------------


class DebugModule : public Module
{
 public:

  JEM_DECLARE_CLASS       ( DebugModule, Module );

  static const char*        TYPE_NAME;


  explicit                  DebugModule

    ( const String&           name  = "debug",
      const Ref<Module>&      child = nullptr );

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

  void                      printHelp

    ( PrintWriter&            out )          const;

  int                       getNoiseLevel () const;

  void                      setNoiseLevel

    ( int                     level );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();


 protected:

  virtual                  ~DebugModule   ();


 private:

  void                      storeVars_

    ( const Properties&       globdat );

  void                      printStatus_

    ( Status                  status );

  bool                      exeCommand_

    ( const String&           cmd,
      const String&           arg,
      const Properties&       globdat );

  void                      exeSetVarCmd_

    ( const String&           arg,
      const Properties&       globdat );

  void                      exeTouchVarCmd_

    ( const String&           arg,
      const Properties&       globdat );

  void                      exeShowDataCmd_

    ( const String&           arg,
      const Properties&       globdat );


 private:

  Ref<Module>               child_;
  Ref<PrintWriter>          output_;
  jem::util::Timer          timer_;

  String                    myContext_;
  int                       noiseLevel_;

};


JIVE_END_PACKAGE( app )

#endif
