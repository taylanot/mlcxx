
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
#include <jive/model/ConstraintsModel.h>
#include <jive/model/DebugModel.h>
#include <jive/model/DummyModel.h>
#include <jive/model/FilterModel.h>
#include <jive/model/LoadScaleModel.h>
#include <jive/model/MatrixModel.h>
#include <jive/model/MatrixFreeModel.h>
#include <jive/model/MPModel.h>
#include <jive/model/MultiModel.h>
#include <jive/model/PointLoadModel.h>
#include <jive/model/declare.h>


JIVE_BEGIN_PACKAGE( model )


//-----------------------------------------------------------------------
//   declareModels
//-----------------------------------------------------------------------


static void declareModels_ ()
{
  ConstraintsModel :: declare ();
  DebugModel       :: declare ();
  DummyModel       :: declare ();
  FilterModel      :: declare ();
  LoadScaleModel   :: declare ();
  MatrixModel      :: declare ();
  MatrixFreeModel  :: declare ();
  MPModel          :: declare ();
  MultiModel       :: declare ();
  PointLoadModel   :: declare ();
}


void declareModels ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareModels_ );
}


JIVE_END_PACKAGE( model )
