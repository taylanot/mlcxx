
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

#ifndef JIVE_FEM_SIMPLEELEMENT_H
#define JIVE_FEM_SIMPLEELEMENT_H

#include <jive/geom/Shape.h>
#include <jive/fem/VirtualElement.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class SimpleElement
//-----------------------------------------------------------------------


class SimpleElement : public VirtualElement
{
 public:

  typedef SimpleElement     Self;
  typedef geom::Shape       Shape;


  inline                    SimpleElement ();

  virtual void              reset         () override;

  virtual void              setShape

    ( Shape*                  shape )        override;

  virtual void              loadData

    ( idx_t                   ielem )        override;

  inline  void              loadDataFast

    ( idx_t                   ielem );


 public:

  idx_t                     ielem;
  Ref<Shape>                shape;
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


inline SimpleElement::SimpleElement ()
{
  ielem     = -1;
  newShape  =  true;
  newShape_ =  true;
}


//-----------------------------------------------------------------------
//   loadDataFast
//-----------------------------------------------------------------------


inline void SimpleElement::loadDataFast ( idx_t ielem )
{
  this->ielem = ielem;
  newShape    = newShape_;
  newShape_   = false;
}


JIVE_END_PACKAGE( fem )

#endif
