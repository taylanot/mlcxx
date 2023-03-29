
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

#ifndef JEM_HDF5_SINK_H
#define JEM_HDF5_SINK_H

#include <jem/base/MemCache.h>
#include <jem/hdf5/Cursor.h>
#include <jem/hdf5/DataType.h>
#include <jem/hdf5/DataSpace.h>


JEM_BEGIN_PACKAGE( hdf5 )


//-----------------------------------------------------------------------
//   class SinkBase
//-----------------------------------------------------------------------


template <class T>

  class SinkBase

{
 public:

  inline                    SinkBase   ();

  inline const DataType&    type       () const noexcept;


 private:

                            SinkBase

    ( const SinkBase&         rhs );

  SinkBase&                 operator =

    ( const SinkBase&         rhs );


 protected:

  DataType                  type_;

};


//-----------------------------------------------------------------------
//   class SinkImp
//-----------------------------------------------------------------------


template <class T, int N>

  class SinkImp : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit                  SinkImp

    ( const Array<T,N>&       data );

  inline                   ~SinkImp    ();

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  void                      flush      () const;
  inline DataSpace          space      () const;


 private:

  const Array<T,N>&         data_;
  T*                        buffer_;

};


//-----------------------------------------------------------------------
//   class SinkImp<T,0>
//-----------------------------------------------------------------------


template <class T>

  class SinkImp<T,0> : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit inline           SinkImp

    ( T&                      data );

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  inline void               flush      () const;
  inline DataSpace          space      () const;


 private:

  T&                        data_;

};


//-----------------------------------------------------------------------
//   class SinkImp<T,1>
//-----------------------------------------------------------------------


template <class T>

  class SinkImp<T,1> : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit                  SinkImp

    ( const Array<T>&         data );

  inline                   ~SinkImp    ();

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  void                      flush      () const;
  inline DataSpace          space      () const;


 private:

  const Array<T>&           data_;
  T*                        buffer_;

};


//-----------------------------------------------------------------------
//   class SinkImp<T,2>
//-----------------------------------------------------------------------


template <class T>

  class SinkImp<T,2> : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit                  SinkImp

    ( const Array<T,2>&       data );

  inline                   ~SinkImp    ();

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  void                      flush      () const;
  inline DataSpace          space      () const;


 private:

  const Array<T,2>&         data_;
  T*                        buffer_;

};


//-----------------------------------------------------------------------
//   class VLSinkImp
//-----------------------------------------------------------------------


template <class T, int N>

  class VLSinkImp : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit inline           VLSinkImp

    ( const Array<T,N>&       data );

  inline                   ~VLSinkImp  ();

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  void                      flush      () const;
  inline const DataSpace&   space      () const;


 private:

  const Array<T,N>&         data_;
  DataSpace                 space_;
  void*                     buffer_;

};


//-----------------------------------------------------------------------
//   class VLSinkImp<T,0>
//-----------------------------------------------------------------------


template <class T>

  class VLSinkImp<T,0> : public SinkBase<T>

{
 public:

  typedef SinkBase<T>       Super;

  explicit inline           VLSinkImp

    ( T&                      data );

  inline                   ~VLSinkImp  ();

  inline void*              addr       () const noexcept;
  inline idx_t              count      () const noexcept;
  inline void               flush      () const;
  inline const DataSpace&   space      () const;


 private:

  T&                        data_;
  DataSpace                 space_;
  void*                     buffer_;

};


//-----------------------------------------------------------------------
//   class Sink
//-----------------------------------------------------------------------


template <class T>

  class Sink : public SinkImp<T,0>

{
 public:

  typedef SinkImp<T,0>      Super;

  explicit inline           Sink

    ( T&                      data );

};


//-----------------------------------------------------------------------
//   class Sink<Array<T,N>>
//-----------------------------------------------------------------------


template <class T, int N>

  class Sink< Array<T,N> > : public SinkImp<T,N>

{
 public:

  typedef SinkImp<T,N>      Super;

  explicit inline           Sink

    ( const Array<T,N>&       data );

};


//-----------------------------------------------------------------------
//   class Sink<Flex<T>>
//-----------------------------------------------------------------------


template <class T>

  class Sink< util::Flex<T> > : public VLSinkImp<util::Flex<T>,0>

{
 public:

  typedef VLSinkImp
    <util::Flex<T>,0>       Super;

  explicit inline           Sink

    ( util::Flex<T>&          data );

};


//-----------------------------------------------------------------------
//   class Sink<Array<Flex<T>,N>>
//-----------------------------------------------------------------------


template <class T, int N>

  class Sink< Array<util::Flex<T>,N> > :

    public VLSinkImp<util::Flex<T>,N>

{
 public:

  typedef VLSinkImp
    <util::Flex<T>,N>       Super;

  explicit inline           Sink

    ( const Array
        <util::Flex<T>,N>&    data );

};


//-----------------------------------------------------------------------
//   class Sink<String>
//-----------------------------------------------------------------------


template <>

  class Sink<String> : public VLSinkImp<String,0>

{
 public:

  typedef
    VLSinkImp<String,0>     Super;

  explicit inline           Sink

    ( String&                 data );

};


//-----------------------------------------------------------------------
//   class Sink<Array<String,N>>
//-----------------------------------------------------------------------


template <int N>

  class Sink< Array<String,N> > : public VLSinkImp<String,N>

{
 public:

  typedef
    VLSinkImp<String,N>     Super;

  explicit inline           Sink

    ( const Array<String,N>&  data );

};





//#######################################################################
//  Implementation
//#######################################################################

//=======================================================================
//   helper functions
//=======================================================================


const void*                 unpack

  ( String&                   data,
    const void*               vlbuf );

const void*                 unpack

  ( ByteFlex&                 data,
    const void*               vlbuf );

const void*                 unpack

  ( ShortFlex&                data,
    const void*               vlbuf );

const void*                 unpack

  ( IntFlex&                  data,
    const void*               vlbuf );

const void*                 unpack

  ( LongFlex&                 data,
    const void*               vlbuf );

const void*                 unpack

  ( LintFlex&                 data,
    const void*               vlbuf );

const void*                 unpack

  ( IdxFlex&                  data,
    const void*               vlbuf );

const void*                 unpack

  ( FloatFlex&                data,
    const void*               vlbuf );

const void*                 unpack

  ( DoubleFlex&               data,
    const void*               vlbuf );


//=======================================================================
//   class SinkBase
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline SinkBase<T>::SinkBase () :

    type_ ( jem::hdf5::typeOf<T>() )

{}


//-----------------------------------------------------------------------
//   type
//-----------------------------------------------------------------------


template <class T>

  inline const DataType& SinkBase<T>::type () const noexcept

{
  return type_;
}


//=======================================================================
//   class SinkImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline SinkImp<T,N>::SinkImp ( const Array<T,N>& data ) :

    data_ ( data )

{
  buffer_ = (T*) MemCache::alloc ( sizeof(T) * data.itemCount() );
}


template <class T, int N>

  inline SinkImp<T,N>::~SinkImp ()

{
  MemCache::dealloc ( buffer_, sizeof(T) * data_.itemCount() );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int N>

  inline void* SinkImp<T,N>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t SinkImp<T,N>::count () const noexcept

{
  return data_.itemCount ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T, int N>

  void SinkImp<T,N>::flush () const

{
  Cursor<N>  c ( data_.shape(), data_.strides() );

  T*         a = data_.addr ();
  const T*   b = buffer_;


  for ( idx_t i = 0; ! c.atEnd(); i++ )
  {
    a[c.next()] = b[i];
  }
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T, int N>

  inline DataSpace SinkImp<T,N>::space () const

{
  return DataSpace::create ( data_.itemCount() );
}


//=======================================================================
//   class SinkImp<T,0>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


template <class T>

  inline SinkImp<T,0>::SinkImp ( T& data ) :

    data_ ( data )

{}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline void* SinkImp<T,0>::addr () const noexcept

{
  return &data_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SinkImp<T,0>::count () const noexcept

{
  return 1;
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T>

  inline void SinkImp<T,0>::flush () const

{}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SinkImp<T,0>::space () const

{
  return DataSpace::create ();
}


//=======================================================================
//   class SinkImp<T,1>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  SinkImp<T,1>::SinkImp ( const Array<T>& data ) :

    data_ ( data )

{
  if ( data.isContiguous() )
  {
    buffer_ = nullptr;
  }
  else
  {
    buffer_ = (T*) MemCache::alloc ( sizeof(T) * data.size() );
  }
}


template <class T>

  inline SinkImp<T,1>::~SinkImp ()

{
  if ( buffer_ )
  {
    MemCache::dealloc ( buffer_, sizeof(T) * data_.size() );
  }
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline void* SinkImp<T,1>::addr () const noexcept

{
  if ( buffer_ )
  {
    return buffer_;
  }
  else
  {
    return data_.addr ();
  }
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SinkImp<T,1>::count () const noexcept

{
  return data_.size ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T>

  void SinkImp<T,1>::flush () const

{
  if ( buffer_ )
  {
    T*           a  = data_.addr   ();
    const T*     b  = buffer_;
    const idx_t  n  = data_.size   ();
    const idx_t  st = data_.stride ();

    for ( idx_t i = 0; i < n; i++ )
    {
      a[i * st] = b[i];
    }
  }
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SinkImp<T,1>::space () const

{
  return DataSpace::create ( data_.size() );
}


//=======================================================================
//   class SinkImp<T,2>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  SinkImp<T,2>::SinkImp ( const Array<T,2>& data ) :

    data_ ( data )

{
  buffer_ = (T*) MemCache::alloc ( sizeof(T) * data.itemCount() );
}


template <class T>

  inline SinkImp<T,2>::~SinkImp ()

{
  MemCache::dealloc ( buffer_, sizeof(T) * data_.itemCount() );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline void* SinkImp<T,2>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t SinkImp<T,2>::count () const noexcept

{
  return data_.itemCount ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T>

  void SinkImp<T,2>::flush () const

{
  T*           a   = data_.addr   ();
  const T*     b   = buffer_;
  const idx_t  m   = data_.size   (0);
  const idx_t  n   = data_.size   (1);
  const idx_t  st0 = data_.stride (0);
  const idx_t  st1 = data_.stride (1);
  idx_t        k   = 0;

  for ( idx_t i = 0; i < m; i++ )
  {
    T*  ai = a + (i * st0);

    for ( idx_t j = 0; j < n; j++, k++ )
    {
      ai[j * st1] = b[k];
    }
  }
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline DataSpace SinkImp<T,2>::space () const

{
  return DataSpace::create ( data_.itemCount() );
}


//=======================================================================
//   class VLSinkImp
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T, int N>

  inline VLSinkImp<T,N>::VLSinkImp ( const Array<T,N>& data ) :

    data_  ( data ),
    space_ ( DataSpace::create( data.itemCount() ) )

{
  buffer_ = vlAlloc ( data.itemCount() );
}


template <class T, int N>

  inline VLSinkImp<T,N>::~VLSinkImp ()

{
  vlReclaim ( buffer_, Super::type_, space_ );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T, int N>

  inline void* VLSinkImp<T,N>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T, int N>

  inline idx_t VLSinkImp<T,N>::count () const noexcept

{
  return data_.itemCount ();
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T, int N>

  void VLSinkImp<T,N>::flush () const

{
  Cursor<N>     c ( data_.shape(), data_.strides() );

  T*            a = data_.addr ();
  const void*   b = (const byte*) buffer_;


  while ( ! c.atEnd() )
  {
    b = unpack ( a[c.next()], b );
  }
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T, int N>

  inline const DataSpace& VLSinkImp<T,N>::space () const

{
  return space_;
}


//=======================================================================
//   class VLSinkImp<T,0>
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


template <class T>

  inline VLSinkImp<T,0>::VLSinkImp ( T& data ) :

    data_  ( data ),
    space_ ( DataSpace::create() )

{
  buffer_ = vlAlloc ( 1 );
}


template <class T>

  inline VLSinkImp<T,0>::~VLSinkImp ()

{
  vlReclaim ( buffer_, Super::type_, space_ );
}


//-----------------------------------------------------------------------
//   addr
//-----------------------------------------------------------------------


template <class T>

  inline void* VLSinkImp<T,0>::addr () const noexcept

{
  return buffer_;
}


//-----------------------------------------------------------------------
//   count
//-----------------------------------------------------------------------


template <class T>

  inline idx_t VLSinkImp<T,0>::count () const noexcept

{
  return 1;
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


template <class T>

  inline void VLSinkImp<T,0>::flush () const

{
  unpack ( data_, buffer_ );
}


//-----------------------------------------------------------------------
//   space
//-----------------------------------------------------------------------


template <class T>

  inline const DataSpace& VLSinkImp<T,0>::space () const

{
  return space_;
}


//=======================================================================
//   class Sink
//=======================================================================


template <class T>

  inline Sink<T>::Sink ( T& data ) :

    Super ( data )

{}


//=======================================================================
//   class Sink<Array<T,N>>
//=======================================================================


template <class T, int N>

  inline Sink< Array<T,N> >::Sink ( const Array<T,N>& data ) :

    Super ( data )

{}


//=======================================================================
//   class Sink<Flex<T>>
//=======================================================================


template <class T>

  inline Sink< util::Flex<T> >::Sink ( util::Flex<T>&  data ) :

    Super ( data )

{}


//=======================================================================
//   class Sink<Array<Flex<T>,N>>
//=======================================================================


template <class T, int N>

  inline Sink< Array<util::Flex<T>,N> >::Sink

  ( const Array<util::Flex<T>,N>&  data ) :

    Super ( data )

{}


//=======================================================================
//   class Sink<String>
//=======================================================================


inline Sink<String>::Sink ( String&  data ) :

  Super ( data )

{}


//=======================================================================
//   class Sink<Array<String,N>>
//=======================================================================


template <int N>

  inline Sink< Array<String,N> >::Sink

  ( const Array<String,N>&  data ) :

    Super ( data )

{}


JEM_END_PACKAGE( hdf5 )

#endif
