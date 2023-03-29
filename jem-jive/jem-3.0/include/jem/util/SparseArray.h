
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

#ifndef JEM_UTIL_SPARSEARRAY_H
#define JEM_UTIL_SPARSEARRAY_H

#include <jem/base/tuple/Tuple.h>
#include <jem/util/FastAllocator.h>
#include <jem/util/HashTableIterator.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class SparseArray
//-----------------------------------------------------------------------


template < class T, int N = 1, class Alloc = FastAllocator >

  class SparseArray

{
 public:

  typedef Tuple<idx_t,N>      Index;
  typedef Tuple<idx_t,N>      Shape;

  class                       Element
  {
   public:

    inline                      Element

      ( const Index&              i,
        const T&                  t );

    inline                      Element

      ( const Element&            rhs );


   public:

    const Index                 index;
    T                           value;

  };

  typedef HashTableNode
    <Element>                 Node;
  typedef Alloc               Allocator;

  typedef HashTableIterator
    < Element >               Iterator;

  typedef HashTableIterator
    < Element,
      const Element&,
      const Element* >        ConstIterator;


  static const int            RANK = N;


                              SparseArray   ();

                              SparseArray

    ( const T&                  zero );

                              SparseArray

    ( const SparseArray&        rhs );

                              SparseArray

    ( SparseArray&&             rhs );

                             ~SparseArray   ();

  inline Iterator             begin         ()       noexcept;
  inline ConstIterator        begin         () const noexcept;
  inline Iterator             end           ()       noexcept;
  inline ConstIterator        end           () const noexcept;

  SparseArray&                operator =

    ( const SparseArray&        rhs );

  SparseArray&                operator =

    ( SparseArray&&             rhs )                noexcept;

  SparseArray&                operator =

    ( T                         rhs );

  void                        swap

    ( SparseArray&              rhs )                noexcept;

  T&                          operator      []

    ( const Index&              i );

  inline T                    operator      []

    ( const Index&              i )            const;

  inline T&                   operator      []

    ( idx_t                     i );

  inline T                    operator      []

    ( idx_t                     i )            const;

  inline T&                   operator      ()

    ( idx_t                     i,
      idx_t                     j );

  inline T                    operator      ()

    ( idx_t                     i,
      idx_t                     j )            const;

  inline T&                   operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k );

  inline T                    operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k )            const;

  inline T&                   operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l );

  inline T                    operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l )            const;

  inline T&                   operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p );

  inline T                    operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p )            const;

  inline T&                   operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p,
      idx_t                     q );

  inline T                    operator      ()

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p,
      idx_t                     q )            const;

  inline T*                   find

    ( const Index&              i );

  inline const T*             find

    ( const Index&              i )            const;

  inline T*                   find

    ( idx_t                     i );

  inline const T*             find

    ( idx_t                     i )            const;

  inline T*                   find

    ( idx_t                     i,
      idx_t                     j );

  inline const T*             find

    ( idx_t                     i,
      idx_t                     j )            const;

  inline T*                   find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k );

  inline const T*             find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k )            const;

  inline T*                   find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l );

  inline const T*             find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l )            const;

  inline T*                   find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p );

  inline const T*             find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p )            const;

  inline T*                   find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p,
      idx_t                     q );

  inline const T*             find

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p,
      idx_t                     q )            const;

  bool                        erase

    ( const Index&              i );

  bool                        erase

    ( idx_t                     i );

  bool                        erase

    ( idx_t                     i,
      idx_t                     j );

  bool                        erase

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k );

  bool                        erase

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l );

  bool                        erase

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p );

  bool                        erase

    ( idx_t                     i,
      idx_t                     j,
      idx_t                     k,
      idx_t                     l,
      idx_t                     p,
      idx_t                     q );

  void                        clear         ();

  void                        reserve

    ( idx_t                     cap );

  void                        trimToSize    ();
  inline idx_t                capacity      () const noexcept;
  void                        reshape       ();

  void                        reshape

    ( const Shape&              sh );

  inline void                 reshape

    ( idx_t                     m );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r );

  inline void                 reshape

    ( idx_t                     m,
      idx_t                     n,
      idx_t                     p,
      idx_t                     q,
      idx_t                     r,
      idx_t                     s );

  inline const Shape&         shape         () const noexcept;
  inline idx_t                size          () const noexcept;

  inline idx_t                size

    ( int                       idim )         const;

  inline const T&             getZero       () const noexcept;

  void                        setZero

    ( const T&                  zero );

  inline idx_t                nonZeroCount  () const noexcept;

  template <class Output>
    void                      printStats

    ( Output&                   out )          const;


 private:

  void                        init_

    ( const T&                  zero,
      const Shape&              sh,
      idx_t                     cap );

  void                        clear_          ();

  void                        copy_

    ( const SparseArray&        rhs );

  inline T*                   find_

    ( const Index&              i )              const;

  void                        rehash_

    ( bool                      prune );

  void                        reserve_

    ( idx_t                     cap );

  inline Node*                newNode_

    ( const Element&            elem,
      lint                      hash );

  inline void                 free_

    ( Node*                     node );


 private:

  Allocator                   allocator_;
  Shape                       shape_;
  Shape                       bounds_;
  Tuple<ulint,N>              params_;
  void**                      table_;
  ulint                       tableBits_;
  ulint                       tableMask_;
  idx_t                       nodeCount_;
  idx_t                       capacity_;
  T*                          zero_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class T, int N, class A>

  inline void                 swap

  ( SparseArray<T,N,A>&         lhs,
    SparseArray<T,N,A>&         rhs ) noexcept;

template <class Input,
          class T, int N, class A>

  void                        decode

  ( Input&                      in,
    SparseArray<T,N,A>&         spa );

template <class Output,
          class T, int N, class A>

  void                        encode

  ( Output&                     out,
    const SparseArray<T,N,A>&   spa );

template <class Output,
          class T, int N, class A>

  void                        print

  ( Output&                     out,
    const SparseArray<T,N,A>&   spa );





//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( util )

#include <jem/util/SparseArray.tcc>

#endif
