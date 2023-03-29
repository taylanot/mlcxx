
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

#ifndef JEM_HDF5_SOURCE_H
#define JEM_HDF5_SOURCE_H

#include <jem/base/MemCache.h>
#include <jem/hdf5/Cursor.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/DataSpace.h>


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class SourceBase
//-----------------------------------------------------------------------


template <class T>

  class SourceBase

{
 public:

  explicit inline           SourceBase   ();

  inline const DataType&    type         () const noexcept;


 private:

                            SourceBase

    ( const SourceBase&       rhs );

  SourceBase&               operator =

    ( const SourceBase&       rhs );


 protected:

  DataType                  type_;

};


//-----------------------------------------------------------------------
//   class SourceImp
//-----------------------------------------------------------------------


template <class T, int N>

  class SourceImp : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit                  SourceImp

    ( const Array<T,N>&       data );

  inline                   ~SourceImp    ();

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  T*                        buffer_;
  idx_t                     count_;

};


//-----------------------------------------------------------------------
//   class SourceImp<T,0>
//-----------------------------------------------------------------------


template <class T>

  class SourceImp<T,0> : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit inline           SourceImp

    ( const T&                data );

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  const T&                  data_;

};


//-----------------------------------------------------------------------
//   class SourceImp<T,1>
//-----------------------------------------------------------------------


template <class T>

  class SourceImp<T,1> : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit                  SourceImp

    ( const Array<T>&         data );

  inline                   ~SourceImp    ();

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  T*                        buffer_;
  idx_t                     count_;
  bool                      allocated_;

};


//-----------------------------------------------------------------------
//   class SourceImp<T,2>
//-----------------------------------------------------------------------


template <class T>

  class SourceImp<T,2> : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit                  SourceImp

    ( const Array<T,2>&       data );

  inline                   ~SourceImp    ();

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  T*                        buffer_;
  idx_t                     count_;

};


//-----------------------------------------------------------------------
//   class VLSourceImp
//-----------------------------------------------------------------------


template <class T, int N>

  class VLSourceImp : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit                  VLSourceImp

    ( const Array<T,N>&       data );

  inline                   ~VLSourceImp  ();

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  void*                     buffer_;
  idx_t                     count_;

};


//-----------------------------------------------------------------------
//   class VLSourceImp<T,0>
//-----------------------------------------------------------------------


template <class T>

  class VLSourceImp<T,0> : public SourceBase<T>

{
 public:

  typedef SourceBase<T>     Super;

  explicit                  VLSourceImp

    ( const T&                data );

  inline                   ~VLSourceImp  ();

  inline const void*        addr         () const noexcept;
  inline idx_t              count        () const noexcept;
  inline DataSpace          space        () const;


 private:

  void*                     buffer_;

};


//-----------------------------------------------------------------------
//   class Source
//-----------------------------------------------------------------------


template <class T>

  class Source : public SourceImp<T,0>

{
 public:

  typedef SourceImp<T,0>    Super;

  explicit inline           Source

    ( const T&                data );

};


//-----------------------------------------------------------------------
//   class Source<Array<T,N>>
//-----------------------------------------------------------------------


template <class T, int N>

  class Source< Array<T,N> > : public SourceImp<T,N>

{
 public:

  typedef SourceImp<T,N>    Super;

  explicit inline           Source

    ( const Array<T,N>&       data );

};


//-----------------------------------------------------------------------
//   class Source<Flex<T>>
//-----------------------------------------------------------------------


template <class T>

  class Source< util::Flex<T> > : public VLSourceImp<util::Flex<T>,0>

{
 public:

  typedef VLSourceImp
    <util::Flex<T>,0>       Super;

  explicit inline           Source

    ( const util::Flex<T>&    data );

};


//-----------------------------------------------------------------------
//   class Source<Array<Flex<T>,N>>
//-----------------------------------------------------------------------


template <class T, int N>

  class Source< Array<util::Flex<T>,N> > :

    public VLSourceImp<util::Flex<T>,N>

{
 public:

  typedef VLSourceImp
    <util::Flex<T>,N>       Super;

  explicit inline           Source

    ( const Array
        <util::Flex<T>,N>&    data );

};


//-----------------------------------------------------------------------
//   class Source<String>
//-----------------------------------------------------------------------


template <>

  class Source<String> : public VLSourceImp<String,0>

{
 public:

  typedef
    VLSourceImp<String,0>   Super;

  explicit inline           Source

    ( const String&           data );

};


//-----------------------------------------------------------------------
//   class Source<Array<String,N>>
//-----------------------------------------------------------------------


template <int N>

  class Source< Array<String,N> > : public VLSourceImp<String,N>

{
 public:

  typedef
    VLSourceImp<String,N>   Super;

  explicit inline           Source

    ( const Array<String,N>&  data );

};




//#######################################################################
//  Implementation
//#######################################################################

//=======================================================================
//   helper functions
//=======================================================================


void*                       pack

  ( void*                     vlbuf,
    const String&             data );

void*                       pack

  ( void*                     vlbuf,
    const ByteFlex&           data );

void*                       pack

  ( void*                     vlbuf,
    const ShortFlex&          data );

void*                       pack

  ( void*                     vlbuf,
    const IntFlex&            data );

void*                       pack

  ( void*                     vlbuf,
    const LongFlex&           data );

void*                       pack

  ( void*                     vlbuf,
    const LintFlex&           data );

void*                       pack

  ( void*                     vlbuf,
    const IdxFlex&            data );

void*                       pack

  ( void*                     vlbuf,
    const FloatFlex&          data );

void*                       pack

  ( void*                     vlbuf,
    const DoubleFlex&         data );


//=======================================================================
//   class SourceBase
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline SourceBase<T>::SourceBase () :

    type_ ( jem::hdf5::typeOf<T>() )

{}


//-----------------------------------------------------------------------
//   type
//-----------------------------------------------------------------------


template <class T>

  inline const DataType& SourceBase<T>::type () const noexcept

{
  return type_;
}


//=======================================================================
//   class SourceImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T, int N>

  SourceImp<T,N>::SourceImp ( const Array<T,N>& data )

{
  Cursor<N>  c ( data.shape(), data.strides() );
  const T*   a = data.addr ();
  T*         b = nullptr;

  count_  = data.itemCount ();
  buffer_ = (T*) MemCache::alloc ( sizeof(T) * count_ );
  b       = buffer_;

  for ( idx_t i = 0; ! c.atEnd(); i++ )
  {
    b[i] = a[c.next()];
  }
}


template <class T, int N>

  inline SourceImp<T,N>::~SourceImp ()

{
  MemCache::dealloc ( buffer_, sizeof(T) * count_ );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int N>

  inline const void* SourceImp<T,N>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t SourceImp<T,N>::count () const noexcept

{
  return count_;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T, int N>

  inline DataSpace SourceImp<T,N>::space () const

{
  return DataSpace::create ( count_ );
}


//=======================================================================
//   class SourceImp<T,0>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline SourceImp<T,0>::SourceImp ( const T& data ) :

    data_ ( data )

{}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline const void* SourceImp<T,0>::addr () const noexcept

{
  return &data_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SourceImp<T,0>::count () const noexcept

{
  return 1;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SourceImp<T,0>::space () const

{
  return DataSpace::create ();
}


//=======================================================================
//   class SourceImp<T,1>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  SourceImp<T,1>::SourceImp ( const Array<T>& data )

{
  count_ = data.size ();

  if ( data.isContiguous() )
  {
    buffer_    = data.addr ();
    allocated_ = false;
  }
  else
  {
    const T*     a  = data.addr   ();
    T*           b  = nullptr;
    const idx_t  n  = count_;
    const idx_t  st = data.stride ();

    buffer_    = (T*) MemCache::alloc ( sizeof(T) * n );
    allocated_ = true;
    b          = buffer_;

    for ( idx_t i = 0; i < n; i++ )
    {
      b[i] = a[i * st];
    }
  }
}


template <class T>

  inline SourceImp<T,1>::~SourceImp ()

{
  if ( allocated_ )
  {
    MemCache::dealloc ( buffer_, sizeof(T) * count_ );
  }
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline const void* SourceImp<T,1>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SourceImp<T,1>::count () const noexcept

{
  return count_;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SourceImp<T,1>::space () const

{
  return DataSpace::create ( count_ );
}


//=======================================================================
//   class SourceImp<T,2>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  SourceImp<T,2>::SourceImp ( const Array<T,2>& data )

{
  const T*     a   = data.addr   ();
  T*           b   = nullptr;
  const idx_t  m   = data.size   (0);
  const idx_t  n   = data.size   (1);
  const idx_t  st0 = data.stride (0);
  const idx_t  st1 = data.stride (1);
  idx_t        k   = 0;

  count_  = data.itemCount ();
  buffer_ = (T*) MemCache::alloc ( sizeof(T) * count_ );
  b       = buffer_;

  for ( idx_t i = 0; i < m; i++ )
  {
    const T*  ai = a + (i * st0);

    for ( idx_t j = 0; j < n; j++, k++ )
    {
      b[k] = ai[j * st1];
    }
  }
}


template <class T>

  inline SourceImp<T,2>::~SourceImp ()

{
  MemCache::dealloc ( buffer_, sizeof(T) * count_ );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline const void* SourceImp<T,2>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SourceImp<T,2>::count () const noexcept

{
  return count_;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SourceImp<T,2>::space () const

{
  return DataSpace::create ( count_ );
}


//=======================================================================
//   class VLSourceImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T, int N>

  VLSourceImp<T,N>::VLSourceImp ( const Array<T,N>& data )

{
  Cursor<N>  c ( data.shape(), data.strides() );

  const T*   a = data.addr ();
  void*      b = nullptr;


  count_  = data.itemCount ();
  buffer_ = vlAlloc ( count_ );
  b       = buffer_;

  try
  {
    while ( ! c.atEnd() )
    {
      b = pack ( b, a[c.next()] );
    }
  }
  catch ( ... )
  {
    vlDealloc ( buffer_, count_, vlElemSize<T>() );
    throw;
  }
}


template <class T, int N>

  inline VLSourceImp<T,N>::~VLSourceImp ()

{
  vlDealloc ( buffer_, count_, vlElemSize<T>() );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int N>

  inline const void* VLSourceImp<T,N>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t VLSourceImp<T,N>::count () const noexcept

{
  return count_;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T, int N>

  inline DataSpace VLSourceImp<T,N>::space () const

{
  return DataSpace::create ( count_ );
}


//=======================================================================
//   class VLSourceImp<T,0>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  inline VLSourceImp<T,0>::VLSourceImp ( const T& data )

{
  buffer_ = vlAlloc ( 1 );

  try
  {
    pack ( buffer_, data );
  }
  catch ( ... )
  {
    vlDealloc ( buffer_, 1, vlElemSize<T>() );
    throw;
  }
}


template <class T>

  inline VLSourceImp<T,0>::~VLSourceImp ()

{
  vlDealloc ( buffer_, 1, vlElemSize<T>() );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline const void* VLSourceImp<T,0>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t VLSourceImp<T,0>::count () const noexcept

{
  return 1;
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace VLSourceImp<T,0>::space () const

{
  return DataSpace::create ();
}


//=======================================================================
//   class Source
//=======================================================================


template <class T>

  inline Source<T>::Source ( const T& data ) :

    Super ( data )

{}


//=======================================================================
//   class Source<Array<T,N>>
//=======================================================================


template <class T, int N>

  inline Source< Array<T,N> >::Source

  ( const Array<T,N>&  data ) :

    Super ( data )

{}


//=======================================================================
//   class Source<Flex<T>>
//=======================================================================


template <class T>

  inline Source< util::Flex<T> >::Source

  ( const util::Flex<T>&  data ) :

    Super ( data )

{}


//=======================================================================
//   class Source<Array<Flex<T>,N>>
//=======================================================================


template <class T, int N>

  inline Source< Array<util::Flex<T>,N> >::Source

  ( const Array<util::Flex<T>,N>&  data ) :

    Super ( data )

{}


//=======================================================================
//   class Source<String>
//=======================================================================


inline Source<String>::Source ( const String& data ) :

  Super ( data )

{}


//=======================================================================
//   class Source<Array<String,N>>
//=======================================================================


template <int N>

  inline Source< Array<String,N> >::Source

  ( const Array<String,N>&  data ) :

    Super ( data )

{}


JEM_END_PACKAGE( hdf5 )

#endif
