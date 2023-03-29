
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */

#ifndef JEM_NUMERIC_SPARSE_SPARSEMATRIX_TCC
#define JEM_NUMERIC_SPARSE_SPARSEMATRIX_TCC

#include <utility>
#include <jem/base/ParseException.h>
#include <jem/base/PrecheckException.h>
#include <jem/numeric/sparse/SparseUtils.h>


JEM_BEGIN_PACKAGE( numeric )


//=======================================================================
//   class SparseMatrix
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


template <class T>

  SparseMatrix<T>::SparseMatrix ()

{}


template <class T>

  SparseMatrix<T>::SparseMatrix ( const Structure& s ) :

    struct_ ( s ),
    values_ ( s.nonZeroCount() )

{}


template <class T>

  SparseMatrix<T>::SparseMatrix ( Structure&& s ) :

    struct_ ( std::move( s ) ),
    values_ ( s.nonZeroCount() )

{}


template <class T>

  SparseMatrix<T>::SparseMatrix

  ( const Structure&  s,
    const Array<T>&   values ) :

    struct_ ( s ),
    values_ ( values )

{
  JEM_PRECHECK2 ( s.nonZeroCount() == values.size(),
                  "SparseMatrix value array has wrong length" );
}


template <class T>

  SparseMatrix<T>::SparseMatrix

  ( Structure&&  s,
    Array<T>&&   values ) :

    struct_ ( std::move( s ) ),
    values_ ( std::move( values ) )

{
  JEM_PRECHECK2 ( s.nonZeroCount() == values.size(),
                  "SparseMatrix value array has wrong length" );
}

template <class T>

  SparseMatrix<T>::SparseMatrix

  ( const Shape&         sh,
    const Array<idx_t>&  offsets,
    const Array<idx_t>&  indices ) :

    struct_ ( sh, offsets, indices ),
    values_ ( indices.size() )

{}


template <class T>

  SparseMatrix<T>::SparseMatrix

  ( const Shape&    sh,
    Array<idx_t>&&  offsets,
    Array<idx_t>&&  indices ) :

    struct_ ( sh, std::move( offsets ), std::move( indices ) ),
    values_ ( indices.size() )

{}


template <class T>

  SparseMatrix<T>::SparseMatrix

  ( const Shape&         sh,
    const Array<idx_t>&  offsets,
    const Array<idx_t>&  indices,
    const Array<T>&      values ) :

    struct_ ( sh, offsets, indices ),
    values_ ( values )

{
  JEM_PRECHECK2 ( indices.size() == values.size(),
                  "SparseMatrix value array has wrong length" );
}


template <class T>

  SparseMatrix<T>::SparseMatrix

  ( const Shape&    sh,
    Array<idx_t>&&  offsets,
    Array<idx_t>&&  indices,
    Array<T>&&      values ) :

    struct_ ( sh, std::move( offsets ), std::move( indices ) ),
    values_ ( std::move( values ) )

{
  JEM_PRECHECK2 ( indices.size() == values.size(),
                  "SparseMatrix value array has wrong length" );
}


template <class T>

  SparseMatrix<T>::SparseMatrix ( const SparseMatrix& rhs )
  noexcept :

    struct_ ( rhs.struct_ ),
    values_ ( rhs.values_ )

{}


template <class T>

  SparseMatrix<T>::SparseMatrix ( SparseMatrix&& rhs ) noexcept :

    struct_ ( std::move( rhs.struct_ ) ),
    values_ ( std::move( rhs.values_ ) )

{}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


template <class T>
  template <class Input>

  void SparseMatrix<T>::readFrom ( Input& in )

{
  decode ( in, struct_, values_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


template <class T>
  template <class Output>

  void SparseMatrix<T>::writeTo ( Output& out ) const

{
  encode ( out, struct_, values_ );
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


template <class T>

  SparseMatrix<T>& SparseMatrix<T>::operator =

  ( const SparseMatrix& rhs ) noexcept

{
  if ( this != &rhs )
  {
    struct_ = rhs.struct_;

    values_.ref ( rhs.values_ );
  }

  return *this;
}


template <class T>

  SparseMatrix<T>& SparseMatrix<T>::operator =

  ( SparseMatrix&& rhs ) noexcept

{
  swap ( rhs );

  return *this;
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  void SparseMatrix<T>::swap ( SparseMatrix& rhs ) noexcept

{
  struct_.swap ( rhs.struct_ );
  values_.swap ( rhs.values_ );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


template <class T>

  SparseMatrix<T> SparseMatrix<T>::clone () const

{
  return SparseMatrix ( struct_.clone(), values_.clone() );
}


//-----------------------------------------------------------------------
//   transpose
//-----------------------------------------------------------------------


template <class T>

  SparseMatrix<T> SparseMatrix<T>::transpose () const

{
  const idx_t   rowCount  = size(0);
  const idx_t   colCount  = size(1);
  const idx_t   nzCount   = nonZeroCount ();

  Array<idx_t>  offsets   = struct_.getRowOffsets    ();
  Array<idx_t>  indices   = struct_.getColumnIndices ();

  Array<idx_t>  trOffsets ( colCount + 1 );
  Array<idx_t>  trIndices ( nzCount );

  Array<T>      trValues  ( nzCount );

  idx_t         first, last;
  idx_t         irow,  jcol;
  idx_t         i;


  // Count the number of entries in each row of the transposed matrix

  trOffsets = 0;
  first     = 0;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    last = offsets[irow + 1];

    for ( i = first; i < last; i++ )
    {
      jcol = indices[i];
      trOffsets[jcol]++;
    }

    first = last;
  }

  SparseUtils::sumOffsets ( trOffsets );

  // Construct the column index array and the value array

  first = 0;

  for ( irow = 0; irow < rowCount; irow++ )
  {
    last = offsets[irow + 1];

    for ( i = first; i < last; i++ )
    {
      jcol                          = indices[i];
      trIndices [ trOffsets[jcol] ] = irow;
      trValues  [ trOffsets[jcol] ] = values_[i];
      trOffsets [ jcol ]++;
    }

    first = last;
  }

  SparseUtils::shiftOffsets ( trOffsets );

  return SparseMatrix ( Shape ( colCount, rowCount ),
                        std::move( trOffsets ),
                        std::move( trIndices ),
                        std::move( trValues ) );
}


//-----------------------------------------------------------------------
//   isValid
//-----------------------------------------------------------------------


template <class T>

  inline bool SparseMatrix<T>::isValid () const noexcept

{
  return struct_.isValid ();
}


//-----------------------------------------------------------------------
//   isSorted
//-----------------------------------------------------------------------


template <class T>

  inline bool SparseMatrix<T>::isSorted () const noexcept

{
  return struct_.isSorted ();
}


//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


template <class T>

  inline bool SparseMatrix<T>::isContiguous () const noexcept

{
  return ( struct_.isContiguous() && values_.isContiguous() );
}


//-----------------------------------------------------------------------
//   getStructure
//-----------------------------------------------------------------------


template <class T>

  inline SparseStructure
  SparseMatrix<T>::getStructure () const noexcept

{
  return struct_;
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SparseMatrix<T>::nonZeroCount () const noexcept

{
  return struct_.nonZeroCount ();
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


template <class T>

  inline typename SparseMatrix<T>::Shape
  SparseMatrix<T>::shape () const noexcept

{
  return struct_.shape ();
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SparseMatrix<T>::size ( int dim ) const

{
  return struct_.size ( dim );
}


//-----------------------------------------------------------------------
//   getRowOffsets
//-----------------------------------------------------------------------


template <class T>

  inline const Array<idx_t>&
  SparseMatrix<T>::getRowOffsets () const noexcept

{
  return struct_.getRowOffsets();
}


//-----------------------------------------------------------------------
//   getColumnIndices
//-----------------------------------------------------------------------


template <class T>

  inline const Array<idx_t>&
  SparseMatrix<T>::getColumnIndices () const noexcept

{
  return struct_.getColumnIndices ();
}


template <class T>

  inline Array<idx_t>
  SparseMatrix<T>::getColumnIndices ( idx_t irow ) const

{
  return struct_.getColumnIndices ( irow );
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


template <class T>

  inline const Array<T>& SparseMatrix<T>::getValues () const noexcept

{
  return values_;
}


template <class T>

  inline Array<T> SparseMatrix<T>::getValues ( idx_t irow ) const

{
  return Array<T> (
    values_,
    slice (
      struct_.getRowOffsets()[irow],
      struct_.getRowOffsets()[irow + 1]
    )
  );
}


//-----------------------------------------------------------------------
//   getOffsetPtr
//-----------------------------------------------------------------------


template <class T>

  inline const idx_t* SparseMatrix<T>::getOffsetPtr () const noexcept

{
  return struct_.getOffsetPtr ();
}


//-----------------------------------------------------------------------
//   getIndexPtr
//-----------------------------------------------------------------------


template <class T>

  inline const idx_t* SparseMatrix<T>::getIndexPtr () const noexcept

{
  return struct_.getIndexPtr ();
}


//-----------------------------------------------------------------------
//   getValuePtr
//-----------------------------------------------------------------------


template <class T>

  inline T* SparseMatrix<T>::getValuePtr () const noexcept

{
  return values_.addr ();
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


template <class T>

  inline void             swap

  ( SparseMatrix<T>&        lhs,
    SparseMatrix<T>&        rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void             decode

  ( Input&                  in,
    SparseMatrix<T>&        a )

{
  a.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void             encode

  ( Output&                 out,
    const SparseMatrix<T>&  a )

{
  a.writeTo( out );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  void                    print

  ( Output&                 out,
    const SparseMatrix<T>&  a )

{
  const idx_t   rowCount = a.size(0);

  Array<idx_t>  offsets  ( a.getRowOffsets() );
  Array<idx_t>  indices  ( a.getColumnIndices() );
  Array<T>      values   ( a.getValues() );

  idx_t         irow, jcol;
  idx_t         i, n;


  for ( irow = 0; irow < rowCount; irow++ )
  {
    i = offsets[irow];
    n = offsets[irow + 1];

    if ( irow > 0 )
    {
      print ( out, '\n' );
    }

    print ( out, "row ", irow, " : " );

    for ( ; i < n; i++ )
    {
      jcol = indices[i];

      print ( out, "( ", jcol, ", ", values[i], " )" );

      if ( i < n - 1 )
      {
        print ( out, ", " );
      }
    }
  }
}


JEM_END_PACKAGE( numeric )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< numeric::SparseMatrix<T> >
//=======================================================================


template <class T>

  class TypeTraits< numeric::SparseMatrix<T> > :

    public BasicTypeTraits

{
 public:

  typedef numeric::
    SparseMatrix<T>     SparseMatrix;


  static const bool     IS_SERIALIZABLE = IsSerializable<T>::VALUE;


  static String         whatis    ()
  {
    return TypeTraits<T>::whatis() + " sparse matrix";
  }


  static String         typeName  ()
  {
    return ( String ( "jem::numeric::SparseMatrix<" ) +
             TypeTraits<T>::typeName ()               +
             String ( '>' ) );
  }


  static SparseMatrix   clone     ( const SparseMatrix& rhs )
  {
    return rhs.clone ();
  }

};


JEM_END_PACKAGE_BASE

#endif
