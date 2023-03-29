
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/Class.h>
#include <jem/base/ClassCastException.h>
#include <jem/gl/Compound.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Compound
//=======================================================================


void Compound::emitVtable_ ()
{}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toCompound
//-----------------------------------------------------------------------


Compound& toCompound ( GfxNode& node )
{
  Compound*  c = dynamic_cast<Compound*> ( &node );

  if ( ! c )
  {
    String  type = node.getClass()->getName ();
    String  name;

    throw ClassCastException (
      JEM_FUNC,
      String::format  (
        "node `%s\' of type `%s\' does not implement "
        "the interface `jem::gl::Compound\'",
        name,
        type
      )
    );
  }

  return *c;
}


//-----------------------------------------------------------------------
//   addToScene
//-----------------------------------------------------------------------


bool                    addToScene

  ( const Ref<GfxNode>&   scene,
    const Ref<GfxNode>&   node )

{
  if ( node == nullptr || node->getParent() )
  {
    return false;
  }

  GfxNode*   n = scene.get ();
  Compound*  c = 0;

  while ( n )
  {
    c = dynamic_cast<Compound*> ( n );

    if ( c )
    {
      c->addChild ( node );

      return true;
    }
  }

  return false;
}


//-----------------------------------------------------------------------
//   eraseFromParent
//-----------------------------------------------------------------------


void eraseFromParent ( const Ref<GfxNode>& node )
{
  if ( ! node )
  {
    return;
  }

  GfxNode*  parent = node->getParent ();

  if ( ! parent )
  {
    return;
  }

  Compound*  c = dynamic_cast<Compound*> ( parent );

  if ( ! c )
  {
    String  nodeName   = node  ->getName ();
    String  parentName = parent->getName ();

    throw RuntimeException (
      JEM_FUNC,
      String::format (
        "failed to remove graphics node `%s\' from its scene "
        "graph because its parent (`%s\') is not a compound node",
        nodeName,
        parentName
      )
    );
  }

  c->eraseChild ( node );
}


JEM_END_PACKAGE( gl )
