
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

#ifndef JEM_NUMERIC_SPARSE_SPARSESTRUCTURE_H
#define JEM_NUMERIC_SPARSE_SPARSESTRUCTURE_H

#include <jem/base/TypeTraits.h>
#include <jem/base/array/Array.h>


namespace jem
{
  namespace io
  {
    class DataInput;
    class DataOutput;
    class TextOutput;
  }
}


JEM_BEGIN_PACKAGE( numeric )


//-----------------------------------------------------------------------
//   class SparseStructure
//-----------------------------------------------------------------------


class SparseStructure
{
 public:

  typedef SparseStructure   Self;
  typedef Tuple<idx_t,2>    Shape;


                            SparseStructure ();

                            SparseStructure

    ( const Shape&            sh,
      const Array<idx_t>&     offsets,
      const Array<idx_t>&     indices );

                            SparseStructure

    ( const Shape&            sh,
      Array<idx_t>&&          offsets,
      Array<idx_t>&&          indices );

                            SparseStructure

    ( const Self&             rhs )                    noexcept;

                            SparseStructure

    ( Self&&                  rhs )                    noexcept;

  void                      readFrom

    ( io::DataInput&          in  );

  void                      writeTo

    ( io::DataOutput&         out )              const;

  Self&                     operator =

    ( const Self&             rhs )                    noexcept;

  Self&                     operator =

    ( Self&&                  rhs )                    noexcept;

  void                      swap

    ( Self&                   rhs )                    noexcept;

  Self                      clone             () const;
  Self                      transpose         () const;
  bool                      isValid           () const noexcept;
  bool                      isSorted          () const noexcept;
  bool                      isSymmetric       () const;
  inline bool               isContiguous      () const noexcept;
  inline idx_t              nonZeroCount      () const noexcept;
  inline Shape              shape             () const noexcept;

  inline idx_t              size

    ( int                     idim )             const;

  inline
    const Array<idx_t>&     getRowOffsets     () const noexcept;
  inline
    const Array<idx_t>&     getColumnIndices  () const noexcept;

  inline  Array<idx_t>      getColumnIndices

    ( idx_t                   irow )             const;

  inline const idx_t*       getOffsetPtr      () const noexcept;
  inline const idx_t*       getIndexPtr       () const noexcept;


 protected:

  Shape                     shape_;
  Array<idx_t>              offsets_;
  Array<idx_t>              indices_;

};


typedef SparseStructure     SparseStruct;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


inline void                 swap

  ( SparseStructure&          lhs,
    SparseStructure&          rhs ) noexcept;

SparseStructure             matmul

  ( const SparseStructure&    a,
    const SparseStructure&    b );

void                        sortRows

  ( SparseStructure&          s );

void                        findEqualRows

  ( const Array<idx_t>&       rmap,
    const SparseStructure&    s );

void                        findSimilarRows

  ( const Array<idx_t>&       rmap,
    const SparseStructure&    s );

template <class Input>

  inline void               decode

  ( Input&                    in,
    SparseStructure&          s );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const SparseStructure&    s );

void                        print

  ( io::TextOutput&           out,
    const SparseStructure&    s );




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class SparseStructure
//=======================================================================

//-----------------------------------------------------------------------
//   isContiguous
//-----------------------------------------------------------------------


inline bool SparseStructure::isContiguous () const noexcept
{
  return (offsets_.stride() + indices_.stride() == 2_idx);
}


//-----------------------------------------------------------------------
//   nonZeroCount
//-----------------------------------------------------------------------


inline idx_t SparseStructure::nonZeroCount () const noexcept
{
  return indices_.size ();
}


//-----------------------------------------------------------------------
//   shape
//-----------------------------------------------------------------------


inline SparseStructure::Shape SparseStructure::shape () const noexcept
{
  return shape_;
}


//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t SparseStructure::size ( int idim ) const
{
  return shape_[idim];
}


//-----------------------------------------------------------------------
//   getRowOffsets
//-----------------------------------------------------------------------


inline const Array<idx_t>&

  SparseStructure::getRowOffsets () const noexcept

{
  return offsets_;
}


//-----------------------------------------------------------------------
//   getColumnIndices
//-----------------------------------------------------------------------


inline const Array<idx_t>&

  SparseStructure::getColumnIndices () const noexcept

{
  return indices_;
}


inline Array<idx_t>

  SparseStructure::getColumnIndices ( idx_t irow ) const

{
  return indices_[slice(offsets_[irow],offsets_[irow + 1])];
}


//-----------------------------------------------------------------------
//   getOffsetPtr
//-----------------------------------------------------------------------


inline const idx_t* SparseStructure::getOffsetPtr () const noexcept
{
  return offsets_.addr ();
}


//-----------------------------------------------------------------------
//   getIndexPtr
//-----------------------------------------------------------------------


inline const idx_t* SparseStructure::getIndexPtr () const noexcept
{
  return indices_.addr ();
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void                 swap

  ( SparseStructure&          lhs,
    SparseStructure&          rhs ) noexcept

{
  lhs.swap ( rhs );
}


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    SparseStructure&          s )

{
  s.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void               encode

  ( Output&                   out,
    const SparseStructure&    s )

{
  s.writeTo ( out );
}


JEM_END_PACKAGE( numeric )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits< numeric::SparseStructure >
//=======================================================================


template <>

  class TypeTraits< numeric::SparseStructure > :

    public BasicTypeTraits

{
 public:

  typedef numeric::
    SparseStructure     SparseStruct;

  static const bool     IS_SERIALIZABLE = true;


  static String         whatis    ()
  {
    return String ( "a sparse matrix structure" );
  }


  static String         typeName  ()
  {
    return String ( "jem::numeric::SparseStructure" );
  }


  static SparseStruct   clone     ( const SparseStruct& rhs )
  {
    return rhs.clone ();
  }

};


JEM_END_PACKAGE_BASE

#endif
