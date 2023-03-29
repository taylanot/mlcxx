
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

#ifndef JIVE_FEM_INTERNALBOUNDARY_H
#define JIVE_FEM_INTERNALBOUNDARY_H

#include <jive/fem/Boundary.h>
#include <jive/fem/VirtualIBoundary.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class InternalBoundary
//-----------------------------------------------------------------------


class InternalBoundary : public Object,
                         public VirtualIBoundary
{
 public:

  JEM_DECLARE_CLASS       ( InternalBoundary, Object );

  static const double       PRECISION;


  explicit                  InternalBoundary

    ( const BoundarySet&      bounds );

                            InternalBoundary

    ( const Ref<Boundary>&    bnd0,
      const Ref<Boundary>&    bnd1 );

  virtual void              reset             ()       override;

  virtual void              setShapes

    ( geom::IShape*           shape0,
      idx_t                   ilocal,
      geom::IShape*           shape1,
      idx_t                   jlocal )                 override;

  virtual void              loadData

    ( idx_t                   ibound,
      idx_t                   ielem,
      idx_t                   jbound,
      idx_t                   jelem  )                 override;

  inline Boundary*          getFirst          () const;
  inline Boundary*          getSecond         () const;

  void                      setPrecision

    ( double                  prec );

  inline double             getPrecision      () const;


 protected:

  virtual                  ~InternalBoundary  ();


 protected:

  Ref<Boundary>             first_;
  Ref<Boundary>             second_;


 private:

  double                    precision_;

  Vector                    xbuf_;

};


//-----------------------------------------------------------------------
//   related typedefs and functions
//-----------------------------------------------------------------------


typedef InternalBoundary    IBoundary;





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getFirst
//-----------------------------------------------------------------------


inline Boundary* InternalBoundary::getFirst () const
{
  return first_.get ();
}


//-----------------------------------------------------------------------
//   getSecond
//-----------------------------------------------------------------------


inline Boundary* InternalBoundary::getSecond () const
{
  return second_.get ();
}


//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------


inline double InternalBoundary::getPrecision () const
{
  return precision_;
}


JIVE_END_PACKAGE( fem )

#endif
