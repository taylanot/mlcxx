
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

#ifndef JIVE_GL_GFXMESH_H
#define JIVE_GL_GFXMESH_H

#include <jem/base/Object.h>
#include <jive/util/Topology.h>
#include <jive/gl/GfxShape.h>


JIVE_BEGIN_PACKAGE( gl )


//-----------------------------------------------------------------------
//   class GfxMesh
//-----------------------------------------------------------------------


class GfxMesh : public Object
{
 public:

  JEM_DECLARE_CLASS     ( GfxMesh, Object );

  typedef jive::
    util::Topology        Topology;


                          GfxMesh       ();

  void                    clear         ();
  void                    check         () const;
  inline idx_t            lineCount     () const;
  inline idx_t            faceCount     () const;
  inline idx_t            vertexCount   () const;
  inline idx_t            xpointCount   () const;
  inline idx_t            firstXPoint   () const;
  idx_t                   addXPoints    ();


 public:

  Topology                lineTopo;
  Topology                faceTopo;

  IdxVector               lineElemMap;
  IdxVector               faceElemMap;
  IdxVector               faceTypeMap;
  IdxVector               vertexNodeMap;

  GfxFaceVector           faceTypes;


 protected:

  virtual                ~GfxMesh       ();


 private:

  void                    checkFailed_

    ( const char*           expr,
      int                   line )         const;


 private:

  idx_t                   xpointCount_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   lineCount
//-----------------------------------------------------------------------


inline idx_t GfxMesh::lineCount () const
{
  return lineElemMap.size ();
}


//-----------------------------------------------------------------------
//   faceCount
//-----------------------------------------------------------------------


inline idx_t GfxMesh::faceCount () const
{
  return faceElemMap.size ();
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


inline idx_t GfxMesh::vertexCount () const
{
  return vertexNodeMap.size ();
}


//-----------------------------------------------------------------------
//   xpointCount
//-----------------------------------------------------------------------


inline idx_t GfxMesh::xpointCount () const
{
  return xpointCount_;
}


//-----------------------------------------------------------------------
//   firstXPoint
//-----------------------------------------------------------------------


inline idx_t GfxMesh::firstXPoint () const
{
  return (vertexNodeMap.size() - xpointCount_);
}


JIVE_END_PACKAGE( gl )

#endif
