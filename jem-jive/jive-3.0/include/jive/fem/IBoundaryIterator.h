
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

#ifndef JIVE_FEM_IBOUNDARYITERATOR_H
#define JIVE_FEM_IBOUNDARYITERATOR_H

#include <jem/io/Serializable.h>
#include <jive/fem/import.h>
#include <jive/fem/ShapeArray.h>
#include <jive/fem/BoundaryGroup.h>
#include <jive/fem/FEIterator.h>


JIVE_BEGIN_PACKAGE( fem )


class VirtualIBoundary;


//-----------------------------------------------------------------------
//   class IBoundaryIterator
//-----------------------------------------------------------------------


class IBoundaryIterator : public FEIterator,
                          public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( IBoundaryIterator, FEIterator );

  typedef geom::BShape      BShape;
  typedef geom::IShape      IShape;


                            IBoundaryIterator ();

                            IBoundaryIterator

    ( const BoundaryGroup&    group,
      const Ref<ShapeTable>&  shapes,
      const String&           context );

                            IBoundaryIterator

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat,
      const String&           context );

  virtual void              readFrom

    ( ObjectInput&            in  )                    override;

  virtual void              writeTo

    ( ObjectOutput&           out )              const override;

  virtual void              resetEvents       ()       override;

  void                      rewind            ();

  bool                      toNext

    ( VirtualIBoundary&       bnd );

  virtual void              checkShapes       ()       override;

  virtual void              setShapeFlags

    ( int                     flags )                  override;

  inline BoundarySet        getBoundaries     () const;
  inline ElementSet         getElements       () const;
  inline NodeSet            getNodes          () const;
  IShapeVector              getShapes         () const;
  IdxVector                 getShapeMap       () const;


 public:

  BoundaryGroup             bndGroup;

  idx_t                     ipos;
  idx_t                     jpos;


 protected:

  virtual                  ~IBoundaryIterator ();


 private:

  void                      init_             ();
  void                      reset_            ();
  void                      update_           ();


 private:

  IShapeVector              shapes_;
  IdxVector                 shapeMap_;

  IdxVector                 iperm_;
  IdxVector                 ibounds_;
  IdxVector                 ielems_;
  IdxVector                 ilocals_;

  idx_t                     inext_;

};


//-----------------------------------------------------------------------
//   related types and functions
//-----------------------------------------------------------------------


typedef IBoundaryIterator   IBndIter;




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getBoundaries
//-----------------------------------------------------------------------


inline BoundarySet IBoundaryIterator::getBoundaries () const
{
  return bndGroup.getBoundaries ();
}


//-----------------------------------------------------------------------
//   getElements
//-----------------------------------------------------------------------


inline ElementSet IBoundaryIterator::getElements () const
{
  return bndGroup.getBoundaries().getElements ();
}


//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------


inline NodeSet IBoundaryIterator::getNodes () const
{
  return bndGroup.getBoundaries().getElements().getNodes ();
}


JIVE_END_PACKAGE( fem )

#endif
