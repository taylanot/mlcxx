
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
#include <jive/gl/declare.h>
#include <jive/gl/ChainPlugin.h>
#include <jive/gl/ElemGroupView.h>
#include <jive/gl/GraphPlugin.h>
#include <jive/gl/MeshReader.h>
#include <jive/gl/MeshView.h>
#include <jive/gl/MeshColorView.h>
#include <jive/gl/MeshNodeView.h>
#include <jive/gl/MarkerManager.h>
#include <jive/gl/NodeDofsView.h>
#include <jive/gl/PaletteManager.h>
#include <jive/gl/PartitionView.h>
#include <jive/gl/PointView.h>
#include <jive/gl/UserconfPlugin.h>


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   declarePlugins
//-----------------------------------------------------------------------


static void declarePlugins_ ()
{
  ChainPlugin    :: declare ();
  ElemGroupView  :: declare ();
  GraphPlugin    :: declare ();
  MeshReader     :: declare ();
  MeshView       :: declare ();
  MeshColorView  :: declare ();
  MeshNodeView   :: declare ();
  MarkerManager  :: declare ();
  NodeDofsView   :: declare ();
  PaletteManager :: declare ();
  PartitionView  :: declare ();
  PointView      :: declare ();
  UserconfPlugin :: declare ();
}


void declarePlugins ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declarePlugins_ );
}


JIVE_END_PACKAGE( gl )
