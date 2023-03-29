
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


#include <jem/base/tuple/operators.h>
#include <jem/gl/Box.h>
#include <jem/gl/MultiNode.h>
#include <jem/gl/Transformable.h>


JEM_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   explode
//-----------------------------------------------------------------------


void          explode

  ( MultiNode&  multi,
    double      scale )

{
  const idx_t     nodeCount = multi.childCount ();

  GfxNode*        node;
  Transformable*  tf;
  Transform       tr;
  Box             box;
  Vertex3d        cm, cn, dx;
  idx_t           n;


  cm = 0.0;
  n  = 0;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    node = multi.getChild ( inode );
    tf   = dynamic_cast<Transformable*> ( node );

    if ( ! tf )
    {
      continue;
    }

    node->getBBox ( box );

    if ( box.isEmpty() )
    {
      continue;
    }

    cm += box.center ();
    n++;
  }

  if ( n == 0 )
  {
    return;
  }

  cm = cm / (GLdouble) n;

  for ( idx_t inode = 0; inode < nodeCount; inode++ )
  {
    node = multi.getChild ( inode );
    tf   = dynamic_cast<Transformable*> ( node );

    if ( ! tf )
    {
      continue;
    }

    node->getBBox ( box );

    if ( box.isEmpty() )
    {
      continue;
    }

    cn = box.center ();
    dx = (GLdouble) scale * (cn - cm);

    tr = tf->getTransform ();

    tr .translate    ( dx[0], dx[1], dx[2] );
    tf->setTransform ( tr );
  }
}


JEM_END_PACKAGE( gl )
