
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
#include <jive/fem/BndgenModule.h>
#include <jive/fem/GroupconvModule.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/MeshpartModule.h>
#include <jive/fem/MPInputModule.h>
#include <jive/fem/PartitionModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/fem/declare.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   declareModules
//-----------------------------------------------------------------------


static void declareModules_ ()
{
  BndgenModule    :: declare ();
  GroupconvModule :: declare ();
  InitModule      :: declare ();
  InputModule     :: declare ();
  MeshpartModule  :: declare ();
  MPInputModule   :: declare ();
  PartitionModule :: declare ();
  ShapeModule     :: declare ();
}


void declareModules ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareModules_ );
}


JIVE_END_PACKAGE( fem )
