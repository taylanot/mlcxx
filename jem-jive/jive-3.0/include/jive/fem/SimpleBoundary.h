
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

#ifndef JIVE_FEM_SIMPLEBOUNDARY_H
#define JIVE_FEM_SIMPLEBOUNDARY_H

#include <jive/geom/BoundaryShape.h>
#include <jive/geom/InternalShape.h>
#include <jive/fem/SimpleElement.h>
#include <jive/fem/VirtualBoundary.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class SimpleBoundary
//-----------------------------------------------------------------------


class SimpleBoundary : public VirtualBoundary
{
 public:

  typedef SimpleBoundary    Self;
  typedef geom::IShape      IShape;
  typedef geom::BShape      BShape;


  inline                    SimpleBoundary  ();

  virtual void              reset           () override;

  virtual void              setShape

    ( IShape*                 shape,
      idx_t                   ilocal )         override;

  virtual void              loadData

    ( idx_t                   ibound,
      idx_t                   ielem  )         override;

  inline  void              loadDataFast

    ( idx_t                   ibound,
      idx_t                   ielem  );


 public:

  SimpleElement             elem;
  idx_t                     ibound;
  idx_t                     ilocal;
  Ref<BShape>               shape;
  bool                      newShape;


 private:

  bool                      newShape_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline SimpleBoundary::SimpleBoundary ()
{
  ibound    = -1;
  ilocal    = -1;
  newShape  =  true;
  newShape_ =  true;
}


//-----------------------------------------------------------------------
//   loadDataFast
//-----------------------------------------------------------------------


inline void SimpleBoundary::loadDataFast

  ( idx_t  ibound,
    idx_t  ielem )

{
  elem.loadDataFast ( ielem );

  this->ibound = ibound;
  newShape     = newShape_;
  newShape_    = false;
}


JIVE_END_PACKAGE( fem )

#endif
