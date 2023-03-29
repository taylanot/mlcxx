
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

#ifndef JIVE_MESH_STRIPMESHER_H
#define JIVE_MESH_STRIPMESHER_H

#include <jem/base/Flags.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/mesh/import.h>


JIVE_BEGIN_PACKAGE( mesh )


class XMesh;


//----------------------------------------------------------------------
//  class StripMesher
//----------------------------------------------------------------------


class StripMesher : public Object
{
 public:

  JEM_DECLARE_CLASS       ( StripMesher, Object );

  enum                      Option
  {
                              WRAP_AROUND      = 1 << 0,
                              NUMBER_CLOCKWISE = 1 << 1
  };

  typedef
    jem::Flags<Option>      Options;


  explicit                  StripMesher

    ( Options                 options  = 0 );

  void                      triangulate3

    ( const XElementSet&      elems,
      const Vector&           xouter,
      const IdxVector&        iouter,
      const Vector&           xinner,
      const IdxVector&        iinner,
      Options                 xtraOpts = 0 );

  void                      triangulate6

    ( const XMesh&            mesh,
      idx_t                   iright,
      const Vector&           xouter,
      const IdxVector&        iouter,
      idx_t                   ileft,
      const Vector&           xinner,
      const IdxVector&        iinner,
      Options                 xtraOpts = 0 );

  inline void               setOption

    ( Option                  option,
      bool                    yesno = true );

  inline void               setOptions

    ( Options                 options );


 protected:

  virtual                  ~StripMesher   ();


 private:

  void                      triangulate3_

    ( const XElementSet&      elems,
      const Vector&           xouter,
      const IdxVector&        iouter,
      const Vector&           xinner,
      const IdxVector&        iinner,
      Options                 options );

  void                      triangulate6_

    ( const XMesh&            mesh,
      idx_t                   iright,
      const Vector&           xouter,
      const IdxVector&        iouter,
      idx_t                   ileft,
      const Vector&           xinner,
      const IdxVector&        iinner,
      Options                 options );

  inline idx_t              addMidNode_

    ( const XMesh&            mesh,
      idx_t                   node0,
      idx_t                   node1 );


 private:

  Options                   options_;

  Vector                    nodePos0_;
  Vector                    nodePos1_;
  IdxVector                 elemNodes_;

};


JEM_DEFINE_FLAG_OPS( StripMesher::Options )


//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


inline void StripMesher::setOption

  ( Option  option,
    bool    yesno )

{
  options_.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


inline void StripMesher::setOptions ( Options options )
{
  options_ = options;
}


JIVE_END_PACKAGE( mesh )

#endif
