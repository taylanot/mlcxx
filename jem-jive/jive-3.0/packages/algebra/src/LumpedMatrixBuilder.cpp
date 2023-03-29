
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


#include <jem/base/assert.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalIndexException.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Event.h>
#include <jem/util/SparseArray.h>
#include <jem/util/FastAllocator.h>
#include <jive/util/error.h>
#include <jive/util/DofSpace.h>
#include <jive/algebra/DiagMatrixObject.h>
#include <jive/algebra/MBuilderParams.h>
#include <jive/algebra/MBuilderFactory.h>
#include <jive/algebra/LumpedMatrixBuilder.h>


JEM_DEFINE_CLASS( jive::algebra::LumpedMatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jem::shape;
using jem::newInstance;
using jem::IllegalIndexException;
using jem::util::SparseArray;


//=======================================================================
//   class LumpedMatrixBuilder::Data_
//=======================================================================


class LumpedMatrixBuilder::Data_ : public jem::Collectable
{
 public:

  typedef Data_             Self;
  typedef
    SparseArray<double>     MBuffer;


  explicit                  Data_

    ( Ref<DiagMatrixObj>      mat );

  inline void               checkRow

    ( idx_t                   irow )           const;

  inline void               checkRowStrict

    ( idx_t                   irow );

  inline void               scale

    ( double                  s );

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline double             getValue

    ( idx_t                   irow,
      idx_t                   jcol  )          const;

  inline idx_t              eraseValue

    ( idx_t                   irow,
      idx_t                   jcol );

  inline void               resize

    ( idx_t                   size );

  inline void               syncSize        ();
  void                      updateMatrix    ();

  void                      getDiagonal

    ( const Vector&           diag )           const;


 public:

  Ref<DofSpace>             dofs;

  Ref<DiagMatrixObj>        output;
  MBuffer                   mbuffer;
  Options                   options;
  double                    multiplier;
  bool                      newValues;


 private:

  void                      valuesChanged_ ();

  void                      checkRowStrict_

    ( idx_t                   irow );

  void                      rowIndexError_

    ( idx_t                   irow )          const;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


LumpedMatrixBuilder::Data_::Data_ ( Ref<DiagMatrixObj> mat ) :

  output ( mat )

{
  using jem::util::connect;

  options    = AUTO_RESHAPE;
  multiplier = 1.0;
  newValues  = true;

  connect ( output->newValuesEvent, this, & Self::valuesChanged_ );
}


//-----------------------------------------------------------------------
//   checkRow & checkRowStrict
//-----------------------------------------------------------------------


inline void LumpedMatrixBuilder::Data_::checkRow ( idx_t irow ) const
{
  if ( irow < 0 )
  {
    rowIndexError_ ( irow );
  }
}


inline void LumpedMatrixBuilder::Data_::checkRowStrict ( idx_t irow )
{
  if ( irow < 0 || irow >= mbuffer.size() )
  {
    checkRowStrict_ ( irow );
  }
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


inline void LumpedMatrixBuilder::Data_::scale ( double s )
{
  MBuffer::Iterator  first = mbuffer.begin ();
  MBuffer::Iterator  last  = mbuffer.end   ();

  for ( ; first != last; ++first )
  {
    first->value *= s;
  }

  newValues = true;
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void LumpedMatrixBuilder::Data_::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  value )

{
  if ( irow == jcol )
  {
    mbuffer[irow] = multiplier * value;
    newValues     = true;
  }
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


inline double LumpedMatrixBuilder::Data_::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  if ( irow == jcol )
  {
    const double* xptr = mbuffer.find ( irow );

    if ( xptr )
    {
      return *xptr;
    }
  }

  return 0.0;
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline idx_t LumpedMatrixBuilder::Data_::eraseValue

  ( idx_t  irow,
    idx_t  jcol )

{
  if ( irow == jcol )
  {
    if ( mbuffer.erase( irow ) )
    {
      newValues = true;

      return 1;
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   resize
//-----------------------------------------------------------------------


inline void LumpedMatrixBuilder::Data_::resize ( idx_t size )
{
  if ( mbuffer.size() != size )
  {
    mbuffer.reshape ( shape( size ) );
    mbuffer.reserve ( size );
  }
}


//-----------------------------------------------------------------------
//   syncSize
//-----------------------------------------------------------------------


inline void LumpedMatrixBuilder::Data_::syncSize ()
{
  if ( dofs )
  {
    resize ( dofs->dofCount() );
  }
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::Data_::updateMatrix ()
{
  using jem::System;

  syncSize ();

  if ( mbuffer.size() != output->size() )
  {
    String  name = output->getName ();
    Vector  diag ( mbuffer.size() );

    print ( System::debug( name ), name,
            " : updating diagonal matrix ...\n" );

    getDiagonal ( diag );

    output->setValues ( diag );
  }
  else if ( newValues )
  {
    String  name = output->getName   ();
    Vector  diag = output->getValues ();

    print ( System::debug( name ), name,
            " : updating diagonal matrix values ...\n" );

    getDiagonal ( diag );

    output->setValues ( diag );
  }

  newValues = false;

  output->resetEvents ();
}


//-----------------------------------------------------------------------
//   getDiagonal
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::Data_::getDiagonal

  ( const Vector&  diag ) const

{
  MBuffer::ConstIterator  first = mbuffer.begin ();
  MBuffer::ConstIterator  last  = mbuffer.end   ();

  double  s = 0.0;

  diag = 0.0;

  for ( ; first != last; ++first )
  {
    diag[first->index[0]] = first->value;
    s                    += first->value;
  }

  if ( jem::Float::isNaN( s ) )
  {
    throw jem::IllegalInputException (
      output->getContext (),
      "invalid matrix element(s): NaN"
    );
  }
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::Data_::valuesChanged_ ()
{
  newValues = true;
}


//-----------------------------------------------------------------------
//   checkRowStrict_
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::Data_::checkRowStrict_ ( idx_t irow )
{
  if ( dofs )
  {
    syncSize ();
  }

  if ( irow < 0 || irow >= mbuffer.size() )
  {
    util::indexError ( output->getContext(), "row",
                       irow, mbuffer.size() );
  }
}


//-----------------------------------------------------------------------
//   rowIndexError_
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::Data_::rowIndexError_ ( idx_t irow ) const
{
  throw jem::IllegalIndexException (
    output->getContext (),
    "negative row index: " + String ( irow )
  );
}


//=======================================================================
//   class LumpedMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LumpedMatrixBuilder::TYPE_NAME = "Lumped";


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


LumpedMatrixBuilder::LumpedMatrixBuilder

  ( const String& name ) :

    Super ( name )

{
  data_ = newInstance<Data_> ( newInstance<DiagMatrixObj>( name ) );
}


LumpedMatrixBuilder::LumpedMatrixBuilder

  ( const String&       name,
    Ref<DiagMatrixObj>  mat )

{
  JEM_PRECHECK2 ( mat, "NULL DiagMatrixObject" );

  data_ = newInstance<Data_> ( mat );
}


LumpedMatrixBuilder::~LumpedMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::clear ()
{
  Data_&       d    = * data_;
  const idx_t  size = d.mbuffer.size ();

  d.mbuffer.clear ();

  if      ( d.dofs )
  {
    d.syncSize ();
  }
  else if ( ! (d.options & AUTO_RESHAPE) )
  {
    d.resize ( size );
  }
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::scale ( double s )
{
  data_->scale ( s );
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setToZero ()
{
  data_->mbuffer   = 0.0;
  data_->newValues = true;

  data_->syncSize ();
}


//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::reserve ( idx_t n )
{
  data_->mbuffer.reserve ( n );
}


//-----------------------------------------------------------------------
//   shapeHint
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::shapeHint

  ( idx_t  rowCount,
    idx_t  colCount )

{
  JEM_PRECHECK2 ( rowCount >= 0 && colCount >= 0,
                  "invalid lumped matrix shape" );

  Data_&  d = * data_;

  d.syncSize ();

  if ( (d.options & AUTO_RESHAPE) && rowCount > d.mbuffer.size() )
  {
    d.resize ( rowCount );
  }
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::trimToSize ()
{
  data_->syncSize           ();
  data_->mbuffer.trimToSize ();
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::updateMatrix ()
{
  data_->updateMatrix ();
}


//-----------------------------------------------------------------------
//   setMultiplier
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setMultiplier ( double x )
{
  data_->multiplier = x;
}


//-----------------------------------------------------------------------
//   getMultiplier
//-----------------------------------------------------------------------


double LumpedMatrixBuilder::getMultiplier () const
{
  return data_->multiplier;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Data_&  d = * data_;

  idx_t   irow, jcol;
  idx_t   i,    j;


  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];

    if ( d.options & AUTO_RESHAPE )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    d.setValue ( irow, jcols[0], values[0] );
  }
  else if ( icount == 1_idx )
  {
    irow = irows[0];

    if ( d.options & AUTO_RESHAPE )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    for ( j = 0; j < jcount; j++ )
    {
      d.setValue ( irow, jcols[j], values[j] );
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[0];

      if ( d.options & AUTO_RESHAPE )
      {
        d.checkRow       ( irow );
      }
      else
      {
        d.checkRowStrict ( irow );
      }

      d.setValue ( irow, jcol, values[i] );
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, values++ )
    {
      irow = irows[i];

      if ( d.options & AUTO_RESHAPE )
      {
        d.checkRow       ( irow );
      }
      else
      {
        d.checkRowStrict ( irow );
      }

      for ( j = 0; j < jcount; j++ )
      {
        d.setValue ( irow, jcols[j], values[j * icount] );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::addData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  values )

{
  Data_&  d = * data_;

  double  tmp;
  idx_t   irow;
  idx_t   i, j;


  d.newValues = true;

  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];

    if ( d.options & AUTO_RESHAPE )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    d.mbuffer[irow] += d.multiplier * values[0];
  }
  else if ( icount == 1_idx )
  {
    irow = irows[0];

    if ( d.options & AUTO_RESHAPE )
    {
      d.checkRow       ( irow );
    }
    else
    {
      d.checkRowStrict ( irow );
    }

    tmp = 0.0;

    for ( j = 0; j < jcount; j++ )
    {
      tmp += values[j];
    }

    d.mbuffer[irow] += d.multiplier * tmp;
  }
  else if ( jcount == 1_idx )
  {
    for ( i = 0; i < icount; i++ )
    {
      irow = irows[0];

      if ( d.options & AUTO_RESHAPE )
      {
        d.checkRow       ( irow );
      }
      else
      {
        d.checkRowStrict ( irow );
      }

      d.mbuffer[irow] += d.multiplier * values[i];
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, values++ )
    {
      irow = irows[i];

      if ( d.options & AUTO_RESHAPE )
      {
        d.checkRow       ( irow );
      }
      else
      {
        d.checkRowStrict ( irow );
      }

      tmp = 0.0;

      for ( j = 0; j < jcount; j++ )
      {
        tmp += values[j * icount];
      }

      d.mbuffer[irow] += d.multiplier * tmp;
    }
  }
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t LumpedMatrixBuilder::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  Data_&  d = * data_;

  idx_t   irow, jcol;
  idx_t   i, j, k;


  k = 0;

  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];

    d.checkRow ( irow );

    k += d.eraseValue ( irow, jcols[0] );
  }
  else if ( icount == 1_idx )
  {
    irow = irows[0];

    d.checkRow ( irow );

    for ( j = 0; j < jcount; j++ )
    {
      k += d.eraseValue ( irow, jcols[j] );
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[0];

      d.checkRow ( irow );

      k += d.eraseValue ( irow, jcol );
    }
  }
  else
  {
    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      d.checkRow ( irow );

      for ( j = 0; j < jcount; j++ )
      {
        k += d.eraseValue ( irow, jcols[j] );
      }
    }
  }

  return k;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  const Data_&  d = * data_;

  idx_t         irow, jcol;
  idx_t         i,    j;


  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];

    d.checkRow ( irow );

    buf[0] = d.getValue ( irow, jcols[0] );
  }
  else if ( icount == 1 )
  {
    irow = irows[0];

    d.checkRow ( irow );

    for ( j = 0; j < jcount; j++ )
    {
      buf[j] = d.getValue ( irow, jcols[j] );
    }
  }
  else if ( jcount == 1 )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[0];

      d.checkRow ( irow );

      buf[i] = d.getValue ( irow, jcol );
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, buf++ )
    {
      irow = irows[i];

      d.checkRow ( irow );

      for ( j = 0; j < jcount; j++ )
      {
        buf[j * icount] = d.getValue ( irow, jcols[j] );
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* LumpedMatrixBuilder::getMatrix () const
{
  return data_->output.get ();
}


//-----------------------------------------------------------------------
//   getDiagMatrix
//-----------------------------------------------------------------------


DiagMatrixObj* LumpedMatrixBuilder::getDiagMatrix () const
{
  return data_->output.get ();
}


//-----------------------------------------------------------------------
//   setSize
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setSize ( idx_t size )
{
  JEM_PRECHECK2 ( size >= 0, "invalid lumped matrix size" );

  Data_&  d = * data_;

  if ( d.dofs )
  {
    size = d.dofs->dofCount ();
  }

  d.resize ( size );
}


//-----------------------------------------------------------------------
//   setOption
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setOption

  ( Option  option,
    bool    yesno )

{
  data_->options.set ( option, yesno );
}


//-----------------------------------------------------------------------
//   setOptions
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::setOptions ( Options options )
{
  Data_&  d = * data_;

  if ( d.dofs )
  {
    options &= (~AUTO_RESHAPE);
  }

  d.options = options;
}


//-----------------------------------------------------------------------
//   getOptions
//-----------------------------------------------------------------------


LumpedMatrixBuilder::Options
  LumpedMatrixBuilder::getOptions () const
{
  return data_->options;
}


//-----------------------------------------------------------------------
//   trackDofSpace
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::trackDofSpace

  ( const Ref<DofSpace>&  dofs )

{
  using jem::util::connect;
  using jem::util::disconnect;

  Data_&   d  = * data_;

  if ( d.dofs == dofs )
  {
    return;
  }

  if ( d.dofs )
  {
    disconnect ( d.dofs->newSizeEvent, this, & Self::newDofCount_ );
  }

  if ( dofs )
  {
    d.options &= (~AUTO_RESHAPE);
    d.dofs     = dofs;

    connect ( dofs->newSizeEvent, this, & Self::newDofCount_ );
  }

  d.syncSize ();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<MBuilder> LumpedMatrixBuilder::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  params,
    const Properties&  globdat )

{
  Ref<DofSpace>  dofs;
  Ref<Self>      mbuilder

    = newInstance<Self> ( name );

  if ( params.find( dofs, MBuilderParams::DOF_SPACE ) )
  {
    mbuilder->trackDofSpace ( dofs );
  }

  return mbuilder;
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::declare ()
{
  MBuilderFactory::declare ( TYPE_NAME,  & makeNew );
  MBuilderFactory::declare ( CLASS_NAME, & makeNew );
}


//-----------------------------------------------------------------------
//   newDofCount_
//-----------------------------------------------------------------------


void LumpedMatrixBuilder::newDofCount_ ()
{
  data_->syncSize ();
}


JIVE_END_PACKAGE( algebra )
