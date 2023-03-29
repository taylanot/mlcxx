
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

#ifndef JEM_UTIL_ARRAYBUFFER_H
#define JEM_UTIL_ARRAYBUFFER_H

#include <initializer_list>
#include <jem/base/array/Array.h>


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class ArrayBuffer
//-----------------------------------------------------------------------


template <class T>

  class ArrayBuffer

{
 public:

  typedef T               ValueType;
  typedef typename
    Array<T>::Iterator    Iterator;
  typedef Iterator        ConstIterator;

  static const float      EXPANSION_FACTOR;


                          ArrayBuffer         ()       noexcept;

  explicit                ArrayBuffer

    ( idx_t                  n );

  explicit                ArrayBuffer

    ( const Array<T>&       rhs );

                          ArrayBuffer

    ( const Array<T>&       buf,
      idx_t                 n );

                          ArrayBuffer

    ( const ArrayBuffer&    rhs );

                          ArrayBuffer

    ( ArrayBuffer&&         rhs )                      noexcept;

  template <class InputIterator>

                          ArrayBuffer

    ( InputIterator         first,
      InputIterator         last );

  template <class U>
                          ArrayBuffer

    ( std::
      initializer_list<U>   list );

  inline Iterator         begin               () const noexcept;
  inline Iterator         end                 () const noexcept;
  inline T&               front               () const;
  inline T&               back                () const;

  ArrayBuffer&            operator =

    ( const ArrayBuffer&    rhs );

  inline ArrayBuffer&     operator =

    ( ArrayBuffer&&         rhs )                      noexcept;

  inline const
    ArrayBuffer&          operator =

    ( T                     rhs )                const;

  void                    swap

    ( ArrayBuffer&          rhs )                      noexcept;

  template <class I>
    inline typename
    array::SliceTraits<T,I>::ResultType

                          operator            []

    ( const I&              i )                  const;

  inline void             pushBack

    ( const T&              item );

  inline void             pushBack

    ( T&&                   item );

  void                    pushBack

    ( T                     item,
      idx_t                 n );

  template <class InputIterator>

    void                  pushBack

    ( InputIterator         first,
      InputIterator         last );

  template <class U>

    inline void           pushBack

    ( std::
      initializer_list<U>   list );

  inline void             popBack             ();

  inline void             popBack

    ( idx_t                 n );

  inline void             clear               ();

  inline void             resize

    ( idx_t                 n );

  inline void             reserve

    ( idx_t                 cap );

  inline void             setExpansionFactor

    ( float                 x );

  inline float            getExpansionFactor  () const noexcept;
  inline void             trimToSize          ();
  inline idx_t            capacity            () const noexcept;
  inline idx_t            size                () const noexcept;
  inline Array<T>         toArray             () const;
  inline const Array<T>&  getBuffer           () const noexcept;
  inline T*               addr                () const noexcept;

  inline T*               addr

    ( idx_t                 offset )             const;


 private:

  void                    grow_

    ( idx_t                 cap );

  inline void             init_

    ( idx_t                 n );


 private:

  Array<T>                buffer_;
  idx_t                   size_;
  float                   expand_;

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input, class T>

  void                    decode

  ( Input&                  in,
    ArrayBuffer<T>&         buf );

template <class Output, class T>

  void                    encode

  ( Output&                 out,
    const ArrayBuffer<T>&   buf );

template <class Output, class T>

  void                    print

  ( Output&                 out,
    const ArrayBuffer<T>&   buf );

template <class T>

  inline void             swap

  ( ArrayBuffer<T>&         lhs,
    ArrayBuffer<T>&         rhs ) noexcept;




//#######################################################################
//   Implementation
//#######################################################################


JEM_END_PACKAGE( util )

#include <jem/util/ArrayBuffer.tcc>

#endif
