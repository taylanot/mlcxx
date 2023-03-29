
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

#ifndef JIVE_ALGEBRA_ABSTRACTMATRIX_H
#define JIVE_ALGEBRA_ABSTRACTMATRIX_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/util/EventSource.h>
#include <jive/algebra/import.h>


JIVE_BEGIN_PACKAGE( algebra )


using jem::util::Event;


//-----------------------------------------------------------------------
//   class MatrixTraits
//-----------------------------------------------------------------------


class MatrixTraits
{
 public:

  static const char*        SYMMETRIC;
  static const char*        DISTRIBUTED;

};


//-----------------------------------------------------------------------
//   class AbstractMatrix
//-----------------------------------------------------------------------


class AbstractMatrix : public NamedObject,
                       public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( AbstractMatrix, NamedObject );

  typedef const void*       ExtensionID;
  typedef
    jem::Tuple<idx_t,2>     Shape;

  Event < Self& >           newValuesEvent;
  Event < Self& >           newStructEvent;


  explicit                  AbstractMatrix

    ( const String&           name = "" );

  virtual void              resetEvents     ()       override;
  virtual String            getContext      () const override;
  virtual Shape             shape           () const = 0;

  inline idx_t              size

    ( int                     idim )           const;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs  )           const = 0;

  virtual void              printTo

    ( PrintWriter&            out )            const;

  virtual void*             getExtByID

    ( ExtensionID             extID )          const;

  template <class Ext>

    inline bool             hasExtension    () const;

  template <class Ext>

    inline Ext*             getExtension    () const;

  virtual bool              hasTrait

    ( const String&           trait )          const;

  inline bool               isSquare        () const;
  inline bool               isSymmetric     () const;
  inline bool               isDistributed   () const;


 protected:

  virtual                  ~AbstractMatrix  ();

  virtual void              emitEvents_

    ( int                     events )               override;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t AbstractMatrix::size ( int idim ) const
{
  return shape()[idim];
}


//-----------------------------------------------------------------------
//   hasExtension
//-----------------------------------------------------------------------


template <class Ext>

  inline bool AbstractMatrix::hasExtension () const

{
  return (getExtByID( Ext::ID ) != 0);
}


//-----------------------------------------------------------------------
//   getExtension
//-----------------------------------------------------------------------


template <class Ext>

  inline Ext* AbstractMatrix::getExtension () const

{
  return static_cast<Ext*> ( getExtByID( Ext::ID ) );
}


//-----------------------------------------------------------------------
//   isSquare
//-----------------------------------------------------------------------


inline bool AbstractMatrix::isSquare () const
{
  const Shape  sh = shape ();

  return (sh[0] == sh[1]);
}


//-----------------------------------------------------------------------
//   isSymmetric
//-----------------------------------------------------------------------


inline bool AbstractMatrix::isSymmetric () const
{
  return hasTrait ( MatrixTraits::SYMMETRIC );
}


//-----------------------------------------------------------------------
//   isDistributed
//-----------------------------------------------------------------------


inline bool AbstractMatrix::isDistributed () const
{
  return hasTrait ( MatrixTraits::DISTRIBUTED );
}


JIVE_END_PACKAGE( algebra )

#endif
