
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

#ifndef JIVE_ALGEBRA_MATRIXBUILDER_H
#define JIVE_ALGEBRA_MATRIXBUILDER_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/algebra/import.h>


JIVE_BEGIN_PACKAGE( algebra )


class AbstractMatrix;


//-----------------------------------------------------------------------
//   class MatrixBuilder
//-----------------------------------------------------------------------


class MatrixBuilder : public NamedObject
{
 public:

  JEM_DECLARE_CLASS       ( MatrixBuilder, NamedObject );


  explicit                  MatrixBuilder

    ( const String&           name = "" );

  virtual String            getContext    () const override;

  virtual void              configure

    ( const Properties&       props );

  virtual void              getConfig

    ( const Properties&       conf )         const;

  virtual void              clear         ()       = 0;

  virtual void              scale

    ( double                  factor )             = 0;

  virtual void              setToZero     ();

  virtual void              reserve

    ( idx_t                   nzCount );

  virtual void              shapeHint

    ( idx_t                   rowCount,
      idx_t                   colCount );

  virtual void              trimToSize    ();
  virtual void              updateMatrix  ()       = 0;

  virtual void              setMultiplier

    ( double                  x )                  = 0;

  virtual double            getMultiplier () const = 0;

  virtual void              setData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )             = 0;

  virtual void              addData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )             = 0;

  virtual idx_t             eraseData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )             = 0;

  virtual void              getData

    ( double*                 buf,
      const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )       const = 0;

  virtual AbstractMatrix*   getMatrix     () const = 0;

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline void               addValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline double             getValue

    ( idx_t                   irow,
      idx_t                   jcol )         const;

  inline bool               eraseValue

    ( idx_t                   irow,
      idx_t                   jcol );

  void                      setBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols,
      const Matrix&           block );

  void                      addBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols,
      const Matrix&           block );

  void                      getBlock

    ( const Matrix&           block,
      const IdxVector&        irows,
      const IdxVector&        jcols )        const;

  idx_t                     eraseBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols );

  void                      setRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols,
      const Vector&           rvals );

  void                      addRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols,
      const Vector&           rvals );

  idx_t                     eraseRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols );

  void                      getRowValues

    ( const Vector&           rvals,
      idx_t                   irow,
      const IdxVector&        jcols )        const;

  void                      setColValues

    ( const IdxVector&        irows,
      idx_t                   jcol,
      const Vector&           cvals );

  void                      addColValues

    ( const IdxVector&        irows,
      idx_t                   jcol,
      const Vector&           cvals );

  idx_t                     eraseColValues

    ( const IdxVector&        irows,
      idx_t                   jcol );

  void                      getColValues

    ( const Vector&           cvals,
      const IdxVector&        irows,
      idx_t                   jcol )         const;


 protected:

  virtual                  ~MatrixBuilder ();

};


//-----------------------------------------------------------------------
//   related functions and types
//-----------------------------------------------------------------------


typedef MatrixBuilder       MBuilder;


Ref<MBuilder>               newMBuilder

  ( const String&             name,
    const Properties&         conf,
    const Properties&         params,
    const Properties&         globdat );

Ref<MBuilder>               newMBuilder

  ( const String&             name,
    const Properties&         conf,
    const Properties&         props,
    const Properties&         params,
    const Properties&         globdat );




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void MatrixBuilder::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  setData ( &irow, 1, &jcol, 1, &value );
}


//-----------------------------------------------------------------------
//   addValue
//-----------------------------------------------------------------------


inline void MatrixBuilder::addValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  addData ( &irow, 1, &jcol, 1, &value );
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


inline double MatrixBuilder::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  double  tmp = 0.0;

  getData ( &tmp, &irow, 1, &jcol, 1 );

  return tmp;
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline bool MatrixBuilder::eraseValue

  ( idx_t  irow,
    idx_t  jcol )

{
  return (eraseData( &irow, 1, &jcol, 1 ) != 0);
}


JIVE_END_PACKAGE( algebra )

#endif
