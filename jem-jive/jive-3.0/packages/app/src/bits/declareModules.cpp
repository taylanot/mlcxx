
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


#include <jem/base/Once.h>
#include <jive/app/declare.h>
#include <jive/app/DebugModule.h>
#include <jive/app/DummyModule.h>
#include <jive/app/ExitModule.h>
#include <jive/app/ChainModule.h>
#include <jive/app/CheckpointModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/GroupgenModule.h>
#include <jive/app/InfoModule.h>
#include <jive/app/InitModule.h>
#include <jive/app/LoadScaleModule.h>
#include <jive/app/LoopModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/SampleModule.h>
#include <jive/app/UserconfModule.h>


JIVE_BEGIN_PACKAGE( app )


//-----------------------------------------------------------------------
//   declareModules
//-----------------------------------------------------------------------


static void declareModules_ ()
{
  ChainModule      :: declare ();
  CheckpointModule :: declare ();
  ControlModule    :: declare ();
  DebugModule      :: declare ();
  DummyModule      :: declare ();
  ExitModule       :: declare ();
  GroupgenModule   :: declare ();
  InfoModule       :: declare ();
  InitModule       :: declare ();
  LoadScaleModule  :: declare ();
  LoopModule       :: declare ();
  OutputModule     :: declare ();
  ReportModule     :: declare ();
  SampleModule     :: declare ();
  UserconfModule   :: declare ();
}


void declareModules ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareModules_ );
}


JIVE_END_PACKAGE( app )
