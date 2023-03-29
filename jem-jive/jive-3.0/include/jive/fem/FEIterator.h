
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

#ifndef JIVE_FEM_FEITERATOR_H
#define JIVE_FEM_FEITERATOR_H

#include <jem/base/Object.h>
#include <jive/fem/import.h>


JIVE_BEGIN_PACKAGE( fem )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class FEIterator
//-----------------------------------------------------------------------


class FEIterator : public Object,
                   public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( FEIterator, Object );

  Event < Self& >           changeEvent;


  virtual void              resetEvents     ()       override;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       conf )           const;

  virtual void              checkShapes     ()       = 0;

  virtual void              setShapeFlags

    ( int                     flags );

  virtual idx_t             setPrecision

    ( idx_t                   prec );

  inline idx_t              getPrecision    () const;
  inline ShapeTable*        getShapeTable   () const;


 protected:

                            FEIterator      ();

                            FEIterator

    ( const Ref<ShapeTable>&  shapes,
      const String&           context );

  virtual                  ~FEIterator      ();

  void                      invalidate_     ();
  void                      checkPrecision_ ();

  void                      readFrom_

    ( ObjectInput&            in );

  virtual void              writeTo_

    ( ObjectOutput&           out  )           const;

  virtual void              emitEvents_

    ( int                     events )               override;


 protected:

  String                    context_;

  Ref<ShapeTable>           shapeTable_;

  int                       shapeFlags_;
  idx_t                     precision_;
  bool                      updated_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


inline idx_t FEIterator::getPrecision () const
{
  return precision_;
}


//-----------------------------------------------------------------------
//   getShapeTable
//-----------------------------------------------------------------------


inline ShapeTable* FEIterator::getShapeTable () const
{
  return shapeTable_.get ();
}


JIVE_END_PACKAGE( fem )

#endif
