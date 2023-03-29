
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
#include <jem/gl/Rect.h>
#include <jem/gl/Camera.h>
#include <jem/gl/GLContext.h>
#include <jem/gl/Transformable.h>


JEM_BEGIN_PACKAGE( gl )


//=======================================================================
//   class Transformable
//=======================================================================


void Transformable::emitVtable_ ()
{}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   toTransformable
//-----------------------------------------------------------------------


Transformable& toTransformable ( GfxNode& node )
{
  Transformable*  t = dynamic_cast<Transformable*> ( &node );

  if ( ! t )
  {
    String  type = node.getClass()->getName ();
    String  name;

    throw ClassCastException (
      JEM_FUNC,
      String::format  (
        "node `%s\' of type `%s\' does not implement "
        "the interface `jem::gl::Transformable\'",
        name,
        type
      )
    );
  }

  return *t;
}


//-----------------------------------------------------------------------
//   getTransformOf
//-----------------------------------------------------------------------


Transform getTransformOf ( const Ref<GfxNode>& leaf )
{
  GfxNode*    node = leaf.get ();

  Transform   tr, tn;


  tr.setIdentity ();

  while ( node )
  {
    Transformable*  trans = dynamic_cast<Transformable*> ( node );

    if ( trans )
    {
      tn = trans->getTransform ();
      tr = matmul ( tn, tr );
    }

    node = node->getParent ();
  }

  return tr;
}


//-----------------------------------------------------------------------
//   getProjectionOf
//-----------------------------------------------------------------------


Transform getProjectionOf ( const Ref<GfxNode>& leaf )
{
  GfxNode*    node = leaf.get ();

  Transform   tr, tn;


  tr.setIdentity ();

  while ( node )
  {
    Transformable*  trans = dynamic_cast<Transformable*> ( node );
    Camera*         cam   = dynamic_cast<Camera*>        ( node );

    if ( trans )
    {
      tn = trans->getTransform ();
      tr = matmul ( tn, tr );
    }

    if ( cam )
    {
      return matmul ( cam->getProjection(), tr );
    }

    node = node->getParent ();
  }

  // No camera found; use an identity projection matrix.

  return tr;
}


//-----------------------------------------------------------------------
//   fromWindowToNode
//-----------------------------------------------------------------------


bool                fromWindowToNode

  ( GLContext&        gtx,
    Vertex3d&         v,
    const GfxNode&    node )

{
  Rect       vport = gtx.getViewport ();

  Transform  p;
  Vertex3d   a;


  if ( vport.isEmpty() )
  {
    return false;
  }

  p = getProjectionOf ( node.getParent() );

  fromWindow ( v, vport );

  if ( ! unproject( a, p, v ) )
  {
    return false;
  }

  v = a;

  return true;
}


JEM_END_PACKAGE( gl )
