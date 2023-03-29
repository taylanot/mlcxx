
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
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/algebra/SparseMatrixObject.h>
#include <jive/algebra/SparseMatrixBuilder.h>
#include "private/search.h"


JEM_DEFINE_CLASS( jive::algebra::SparseMatrixBuilder );


JIVE_BEGIN_PACKAGE( algebra )


using jem::IllegalInputException;


//=======================================================================
//   class SparseMatrixBuilder
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


SparseMatrixBuilder::SparseMatrixBuilder

  ( const String&        name,
    Ref<SparseMatrixObj> mat ) :

    Super   ( name ),
    output_ ( mat  )

{
  if ( ! output_ )
  {
    output_ = jem::newInstance<SparseMatrixObj> ( name );
  }

  init_ ();
}


SparseMatrixBuilder::~SparseMatrixBuilder ()
{}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void SparseMatrixBuilder::clear ()
{
  setToZero ();
}


//-----------------------------------------------------------------------
//   scale
//-----------------------------------------------------------------------


void SparseMatrixBuilder::scale ( double s )
{
  values_   *= s;
  newValues_ = true;
}


//-----------------------------------------------------------------------
//   setToZero
//-----------------------------------------------------------------------


void SparseMatrixBuilder::setToZero ()
{
  values_    = 0.0;
  newValues_ = true;
}


//-----------------------------------------------------------------------
//   updateMatrix
//-----------------------------------------------------------------------


void SparseMatrixBuilder::updateMatrix ()
{
  using jem::Float;
  using jem::System;

  if ( ! (newStruct_ || newValues_) )
  {
    return;
  }

  if ( Float::isNaN( jem::sum( values_ ) ) )
  {
    throw IllegalInputException (
      getContext (),
      "invalid matrix element(s): NaN"
    );
  }

  if      ( newStruct_ )
  {
    print ( System::debug( myName_ ), myName_,
            " : updating sparse matrix structure ...\n" );

    output_->setMatrix (
      SparseMatrix ( jem::shape ( rowCount_, colCount_ ),
                     rowOffsets_, colIndices_, values_ )
    );
  }
  else if ( newValues_ )
  {
    print ( System::debug( myName_ ), myName_,
            " : updating sparse matrix values ...\n" );

    output_->setValues ( values_ );
  }

  newStruct_ = false;
  newValues_ = false;

  output_->resetEvents ();
}


//-----------------------------------------------------------------------
//   setMultiplier
//-----------------------------------------------------------------------


void SparseMatrixBuilder::setMultiplier ( double x )
{
  multiplier_ = x;
}


//-----------------------------------------------------------------------
//   getMultiplier
//-----------------------------------------------------------------------


double SparseMatrixBuilder::getMultiplier () const
{
  return multiplier_;
}


//-----------------------------------------------------------------------
//   setData
//-----------------------------------------------------------------------


void SparseMatrixBuilder::setData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  data )

{
  const idx_t*  JEM_RESTRICT colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT rowOffsets = rowOffsets_.addr ();

  const double  scale = multiplier_;

  idx_t         first, last;
  idx_t         irow,  jcol;
  idx_t         i, j, k;


  newValues_ = true;

  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];
    jcol = jcols[0];

    checkRowIndex_   ( irow );

    k = binarySearch ( jcol,
                       colIndices,
                       rowOffsets[irow],
                       rowOffsets[irow + 1] );

    if ( k < 0 )
    {
      noSuchValueError_ ( irow, jcol );
    }

    values_[k] = scale * data[0];
  }
  else if ( icount == 1_idx )
  {
    irow  = irows[0];

    checkRowIndex_ ( irow );

    first = rowOffsets[irow];
    last  = rowOffsets[irow + 1];

    for ( j = 0; j < jcount; )
    {
      k = binarySearch ( jcols[j], colIndices, first, last );

      if ( k < 0 )
      {
        noSuchValueError_ ( irow, jcols[j] );
      }

      values_[k] = scale * data[j];

      for ( j++, k++;
            j < jcount && k < last && jcols[j] == colIndices[k];
            j++, k++ )
      {
        values_[k] = scale * data[j];
      }
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      checkRowIndex_   ( irow );

      k = binarySearch ( jcol,
                         colIndices,
                         rowOffsets[irow],
                         rowOffsets[irow + 1] );

      if ( k < 0 )
      {
        noSuchValueError_ ( irow, jcol );
      }

      values_[k] = scale * data[i];
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, data++ )
    {
      irow  = irows[i];
      first = rowOffsets[irow];
      last  = rowOffsets[irow + 1];

      checkRowIndex_ ( irow );

      for ( j = 0; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k < 0 )
        {
          noSuchValueError_ ( irow, jcols[j] );
        }

        values_[k] = scale * data[j * icount];

        for ( j++, k++;
              j < jcount && k < last && jcols[j] == colIndices[k];
              j++, k++ )
        {
          values_[k] = scale * data[j * icount];
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------


void SparseMatrixBuilder::addData

  ( const idx_t*   irows,
    idx_t          icount,
    const idx_t*   jcols,
    idx_t          jcount,
    const double*  data )

{
  const idx_t*  JEM_RESTRICT colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT rowOffsets = rowOffsets_.addr ();

  const double  scale = multiplier_;

  idx_t         first, last;
  idx_t         irow,  jcol;
  idx_t         i, j, k;


  newValues_ = true;

  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];
    jcol = jcols[0];

    checkRowIndex_   ( irow );

    k = binarySearch ( jcol,
                       colIndices,
                       rowOffsets[irow],
                       rowOffsets[irow + 1] );

    if ( k < 0 )
    {
      noSuchValueError_ ( irow, jcol );
    }

    values_[k] += scale * data[0];
  }
  else if ( icount == 1_idx )
  {
    irow  = irows[0];

    checkRowIndex_ ( irow );

    first = rowOffsets[irow];
    last  = rowOffsets[irow + 1];

    for ( j = 0; j < jcount; )
    {
      k = binarySearch ( jcols[j], colIndices, first, last );

      if ( k < 0 )
      {
        noSuchValueError_ ( irow, jcols[j] );
      }

      values_[k] += scale * data[j];

      for ( j++, k++;
            j < jcount && k < last && jcols[j] == colIndices[k];
            j++, k++ )
      {
        values_[k] += scale * data[j];
      }
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      checkRowIndex_   ( irow );

      k = binarySearch ( jcol,
                         colIndices,
                         rowOffsets[irow],
                         rowOffsets[irow + 1] );

      if ( k < 0 )
      {
        noSuchValueError_ ( irow, jcol );
      }

      values_[k] += scale * data[i];
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, data++ )
    {
      irow  = irows[i];
      first = rowOffsets[irow];
      last  = rowOffsets[irow + 1];

      checkRowIndex_ ( irow );

      for ( j = 0; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k < 0 )
        {
          noSuchValueError_ ( irow, jcols[j] );
        }

        values_[k] += scale * data[j * icount];

        for ( j++, k++;
              j < jcount && k < last && jcols[j] == colIndices[k];
              j++, k++ )
        {
          values_[k] += scale * data[j * icount];
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   eraseData
//-----------------------------------------------------------------------


idx_t SparseMatrixBuilder::eraseData

  ( const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount )

{
  const idx_t*  JEM_RESTRICT colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT rowOffsets = rowOffsets_.addr ();

  idx_t         first, last;
  idx_t         irow,  jcol;
  idx_t         i, j, k;


  newValues_ = true;

  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];
    jcol = jcols[0];

    k = binarySearch ( jcol,
                       colIndices,
                       rowOffsets[irow],
                       rowOffsets[irow + 1] );

    if ( k >= 0 )
    {
      values_[k] = 0.0;
    }
  }
  else if ( icount == 1_idx )
  {
    irow  = irows[0];

    checkRowIndex_ ( irow );

    first = rowOffsets[irow];
    last  = rowOffsets[irow + 1];

    for ( j = 0; j < jcount; )
    {
      k = binarySearch ( jcols[j], colIndices, first, last );

      if ( k < 0 )
      {
        j++;
      }
      else
      {
        values_[k] = 0.0;

        for ( j++, k++;
              j < jcount && k < last && jcols[j] == colIndices[k];
              j++, k++ )
        {
          values_[k] = 0.0;
        }
      }
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      checkRowIndex_   ( irow );

      k = binarySearch ( jcol,
                         colIndices,
                         rowOffsets[irow],
                         rowOffsets[irow + 1] );

      if ( k >= 0 )
      {
        values_[k] = 0.0;
      }
    }
  }
  else
  {
    for ( i = 0; i < icount; i++ )
    {
      irow  = irows[i];
      first = rowOffsets[irow];
      last  = rowOffsets[irow + 1];

      checkRowIndex_ ( irow );

      for ( j = 0; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k < 0 )
        {
          j++;
        }
        else
        {
          values_[k] = 0.0;

          for ( j++, k++;
                j < jcount && k < last && jcols[j] == colIndices[k];
                j++, k++ )
          {
            values_[k] = 0.0;
          }
        }
      }
    }
  }

  return 0;
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


void SparseMatrixBuilder::getData

  ( double*       buf,
    const idx_t*  irows,
    idx_t         icount,
    const idx_t*  jcols,
    idx_t         jcount ) const

{
  const idx_t*  JEM_RESTRICT colIndices = colIndices_.addr ();
  const idx_t*  JEM_RESTRICT rowOffsets = rowOffsets_.addr ();

  idx_t         first, last;
  idx_t         irow,  jcol;
  idx_t         i, j, k;


  if      ( icount * jcount == 1_idx )
  {
    irow = irows[0];
    jcol = jcols[0];

    checkRowIndex_   ( irow );

    k = binarySearch ( jcol,
                       colIndices,
                       rowOffsets[irow],
                       rowOffsets[irow + 1] );

    if ( k < 0 )
    {
      buf[0] = 0.0;
    }
    else
    {
      buf[0] = values_[k];
    }
  }
  else if ( icount == 1_idx )
  {
    irow  = irows[0];

    checkRowIndex_ ( irow );

    first = rowOffsets[irow];

    last  = rowOffsets[irow + 1];

    for ( j = 0; j < jcount; )
    {
      k = binarySearch ( jcols[j], colIndices, first, last );

      if ( k < 0 )
      {
        buf[j] = 0.0;

        j++;
      }
      else
      {
        buf[j] = values_[k];

        for ( j++, k++;
              j < jcount && k < last && jcols[j] == colIndices[k];
              j++, k++ )
        {
          buf[j] = values_[k];
        }
      }
    }
  }
  else if ( jcount == 1_idx )
  {
    jcol = jcols[0];

    for ( i = 0; i < icount; i++ )
    {
      irow = irows[i];

      checkRowIndex_   ( irow );

      k = binarySearch ( jcol,
                         colIndices,
                         rowOffsets[irow],
                         rowOffsets[irow + 1] );

      if ( k < 0 )
      {
        buf[i] = 0.0;
      }
      else
      {
        buf[i] = values_[k];
      }
    }
  }
  else
  {
    for ( i = 0; i < icount; i++, buf++ )
    {
      irow  = irows[i];
      first = rowOffsets[irow];
      last  = rowOffsets[irow + 1];

      checkRowIndex_ ( irow );

      for ( j = 0; j < jcount; )
      {
        k = binarySearch ( jcols[j], colIndices, first, last );

        if ( k < 0 )
        {
          buf[j * icount] = 0.0;

          j++;
        }
        else
        {
          buf[j * icount] = values_[k];

          for ( j++, k++;
                j < jcount && k < last && jcols[j] == colIndices[k];
                j++, k++ )
          {
            buf[j * icount] = values_[k];
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getMatrix
//-----------------------------------------------------------------------


AbstractMatrix* SparseMatrixBuilder::getMatrix () const
{
  return output_.get ();
}


//-----------------------------------------------------------------------
//   getSparseMatrix
//-----------------------------------------------------------------------


SparseMatrixObj* SparseMatrixBuilder::getSparseMatrix () const
{
  return output_.get ();
}


//-----------------------------------------------------------------------
//   setStructure
//-----------------------------------------------------------------------


void SparseMatrixBuilder::setStructure ( const SparseStruct& st )
{
  rowOffsets_.ref ( st.getRowOffsets()    );
  colIndices_.ref ( st.getColumnIndices() );

  if ( ! rowOffsets_.isContiguous() )
  {
    rowOffsets_.ref ( rowOffsets_.clone() );
  }

  if ( ! colIndices_.isContiguous() )
  {
    colIndices_.ref ( colIndices_.clone() );
  }

  rowCount_  = st.size(0);
  colCount_  = st.size(1);
  newStruct_ = true;

  if ( values_.size() != colIndices_.size() )
  {
    values_.ref ( Vector( colIndices_.size() ) );
  }

  values_ = 0.0;

  output_->setMatrix ( SparseMatrix() );
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void SparseMatrixBuilder::init_ ()
{
  using jem::util::connect;

  SparseMatrixObj::Shape  sh = output_->shape ();

  rowOffsets_.ref ( output_->getRowOffsets()    );
  colIndices_.ref ( output_->getColumnIndices() );
  values_    .ref ( output_->getValues()        );

  if ( ! rowOffsets_.isContiguous() )
  {
    rowOffsets_.ref ( rowOffsets_.clone() );
  }

  if ( ! colIndices_.isContiguous() )
  {
    colIndices_.ref ( colIndices_.clone() );
  }

  multiplier_ = 1.0;
  rowCount_   = sh[0];
  colCount_   = sh[1];
  newValues_  = false;
  newStruct_  = false;

  connect ( output_->newValuesEvent, this, & Self::valuesChanged_ );
  connect ( output_->newStructEvent, this, & Self::structChanged_ );
}


//-----------------------------------------------------------------------
//   valuesChanged_
//-----------------------------------------------------------------------


void SparseMatrixBuilder::valuesChanged_ ()
{
  newValues_ = true;
}


//-----------------------------------------------------------------------
//   structChanged_
//-----------------------------------------------------------------------


void SparseMatrixBuilder::structChanged_ ()
{
  newStruct_ = true;
}


//-----------------------------------------------------------------------
//   checkRowIndex_
//-----------------------------------------------------------------------


inline void SparseMatrixBuilder::checkRowIndex_ ( idx_t irow ) const
{
  if ( irow < 0 || irow >= rowCount_ )
  {
    util::indexError ( getContext (),
                       "row index", irow, rowCount_ );
  }
}


//-----------------------------------------------------------------------
//   noSuchValueError_
//-----------------------------------------------------------------------


void SparseMatrixBuilder::noSuchValueError_

  ( idx_t  irow,
    idx_t  jcol ) const

{
  throw IllegalInputException (
    getContext     (),
    String::format (
      "illegal sparse matrix element: (%d,%d)", irow, jcol
    )
  );
}


JIVE_END_PACKAGE( algebra )
