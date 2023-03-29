
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

#ifndef JEM_BASE_OBJECT_H
#define JEM_BASE_OBJECT_H

#include <jem/base/Ref.h>
#include <jem/base/String.h>


JEM_BEGIN_PACKAGE_BASE


class Class;


//-----------------------------------------------------------------------
//   class Object
//-----------------------------------------------------------------------


class Object : public Collectable
{
 public:

  typedef Object          Self;
  typedef Object          DeclaringClass;

  static const char*      CLASS_NAME;


  static  Class*          makeClass     ();
  static  Class*          getType       ();
  virtual Class*          getClass      () const;

  virtual String          toString      () const;
  virtual lint            hashValue     () const;

  virtual bool            equals

    ( const Ref<Object>&    rhs )          const;

  Ref<Object>             clone         () const;


 protected:

  inline                  Object        ();
  virtual inline         ~Object        ();

};


//-----------------------------------------------------------------------
//   related macros
//-----------------------------------------------------------------------


#undef  JEM_DECLARE_CLASS
#define JEM_DECLARE_CLASS( self, super )        \
                                                \
  typedef self            Self;                 \
  typedef super           Super;                \
  typedef self            DeclaringClass;       \
                                                \
  static const char*      CLASS_NAME;           \
                                                \
  static  jem::Class*     getType     ();       \
  static  jem::Class*     makeClass   ();       \
  virtual jem::Class*     getClass    () const override


//-----------------------------------------------------------------------
//   class Ref<Object>
//-----------------------------------------------------------------------


template <>

  class Ref<Object>

{
 public:

  enum                    TypeID
  {
                            NIL_TYPE,
                            BOOL_TYPE,
                            CHAR_TYPE,
                            STRING_TYPE,
                            INT_TYPE,
                            FLOAT_TYPE,
                            OBJECT_TYPE
  };


  inline                  Ref           ()       noexcept;

  inline                  Ref

    ( const Nil&            rhs )                noexcept;

  inline                  Ref

    ( std::nullptr_t        rhs )                noexcept;

  inline                  Ref

    ( Object*               rhs )                noexcept;

                          Ref

    ( const Ref&            rhs )                noexcept;

  template <class T>
    inline                Ref

    ( const Ref<T>&         rhs )                noexcept;

                          Ref

    ( Ref&&                 rhs )                noexcept;

  inline                  Ref

    ( bool                  rhs,
      const RawRef&         raw )                noexcept;

  inline                  Ref

    ( char                  rhs,
      const RawRef&         raw )                noexcept;

  inline                  Ref

    ( lint                  rhs,
      const RawRef&         raw )                noexcept;

  inline                  Ref

    ( double                rhs,
      const RawRef&         raw )                noexcept;

                          Ref

    ( const String&         rhs,
      const RawRef&         raw );

  inline                  Ref

    ( Object&               rhs,
      Collectable&          base )               noexcept;

  inline                  Ref

    ( Object&               rhs,
      Collectable&          base,
      const RawRef&         raw )                noexcept;

  inline                 ~Ref           ()       noexcept;

  Ref&                    operator =

    ( const Ref&            rhs )                noexcept;

  template <class T>
    inline Ref&           operator =

    ( const Ref<T>&         rhs )                noexcept;

  inline Ref&             operator =

    ( Ref&&                 rhs )                noexcept;

  Ref&                    operator =

    ( Object*               rhs )                noexcept;

  inline Ref&             operator =

    ( const Nil&            rhs )                noexcept;

  Ref&                    operator =

    ( std::nullptr_t        rhs )                noexcept;

  inline bool             operator ==

    ( const Ref&            rhs )          const noexcept;

  template <class T>
    inline bool           operator ==

    ( const Ref<T>&         rhs )          const noexcept;

  template <class T>
    inline bool           operator ==

    ( const T*              rhs )          const noexcept;

  inline bool             operator ==

    ( const Nil&            rhs )          const noexcept;

  inline bool             operator ==

    ( std::nullptr_t        rhs )          const noexcept;

  inline Object*          operator ->   () const;
  inline Object&          operator  *   () const;
  constexpr explicit      operator bool () const noexcept;
  constexpr inline bool   operator  !   () const noexcept;

  void                    swap

    ( Ref&                  rhs )                noexcept;

  inline Object*          get           () const;
  inline Collectable*     getBase       () const;
  inline void             release       ()       noexcept;

  inline TypeID           getType       () const noexcept;
  Class*                  getClass      () const noexcept;

  String                  toString      () const;
  lint                    hashValue     () const;

  bool                    equals

    ( const Ref&            rhs )          const;

  Ref                     clone         () const;

  inline bool             getBool       () const;
  inline char             getChar       () const;
  inline lint             getInt        () const;
  inline double           getFloat      () const;
  inline String           getString     () const;
  inline Object&          getObject     () const;

  static Object*          dynamicCast

    ( Object&               rhs,
      Class*                target );


 private:

  void                    toObject_     () const;

  inline bool             checkAddr_    () const;
  static void             nullError_    ();


 private:

  TypeID                  type_;

  union                   Data_
  {
    lint                    ivalue;
    double                  xvalue;
    Object*                 object;
    union
    {
      char                  string[TinyString::MAXLEN];
      void*                 strptr;
    };
  };

  Data_                   data_;

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Object
//=======================================================================


inline Object::Object  ()
{}


inline Object::~Object ()
{}


//=======================================================================
//   class Ref<Object>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


inline Ref<Object>::Ref () noexcept
{
  type_        = NIL_TYPE;
  data_.object = nullptr;
}


inline Ref<Object>::Ref ( const Nil& ) noexcept
{
  type_        = NIL_TYPE;
  data_.object = nullptr;
}


inline Ref<Object>::Ref ( std::nullptr_t ) noexcept
{
  type_        = NIL_TYPE;
  data_.object = nullptr;
}


inline Ref<Object>::Ref ( Object* rhs ) noexcept
{
  data_.object = rhs;

  if ( rhs )
  {
    type_ = OBJECT_TYPE;

    Collectable::addRef ( *rhs );
  }
  else
  {
    type_ = NIL_TYPE;
  }
}


template <class T>

  inline Ref<Object>::Ref ( const Ref<T>& rhs ) noexcept

{
  data_.object = rhs.get ();

  if ( data_.object )
  {
    type_ = OBJECT_TYPE;

    Collectable::addRef ( *data_.object );
  }
  else
  {
    type_ = NIL_TYPE;
  }
}


inline Ref<Object>::Ref

  ( bool           rhs,
    const RawRef&  raw ) noexcept

{
  type_        = BOOL_TYPE;
  data_.ivalue = rhs ? 1_lint : 0_lint;
}


inline Ref<Object>::Ref

  ( char           rhs,
    const RawRef&  raw ) noexcept

{
  type_        = CHAR_TYPE;
  data_.ivalue = rhs;
}


inline Ref<Object>::Ref

  ( lint           rhs,
    const RawRef&  raw ) noexcept

{
  type_        = INT_TYPE;
  data_.ivalue = rhs;
}


inline Ref<Object>::Ref

  ( double         rhs,
    const RawRef&  raw ) noexcept

{
  type_        = FLOAT_TYPE;
  data_.xvalue = rhs;
}


inline Ref<Object>::Ref

  ( Object&       rhs,
    Collectable&  base ) noexcept

{
  type_        = OBJECT_TYPE;
  data_.object = & rhs;

  Collectable::addRef ( rhs );
}


inline Ref<Object>::Ref

  ( Object&        rhs,
    Collectable&   base,
    const RawRef&  raw ) noexcept

{
  type_        = OBJECT_TYPE;
  data_.object = & rhs;
}


inline Ref<Object>::~Ref () noexcept
{
  if ( type_ == OBJECT_TYPE )
  {
    Collectable::delRef ( *data_.object );
  }
}


//-----------------------------------------------------------------------
//   assignment operators
//-----------------------------------------------------------------------


inline Ref<Object>& Ref<Object>::operator = ( Ref&& rhs ) noexcept
{
  swap ( rhs );

  return *this;
}


template <class T>

  inline Ref<Object>&
  Ref<Object>::operator = ( const Ref<T>& rhs ) noexcept

{
  return operator = ( rhs.get() );
}


inline Ref<Object>&
  Ref<Object>::operator = ( const Nil& rhs ) noexcept
{
  return operator = ( nullptr );
}


//-----------------------------------------------------------------------
//   equality operators
//-----------------------------------------------------------------------


inline bool
  Ref<Object>::operator == ( const Ref& rhs ) const noexcept
{
  if ( type_ == OBJECT_TYPE && rhs.type_ == OBJECT_TYPE )
  {
    return (data_.object == rhs.data_.object);
  }
  else
  {
    return (type_ == NIL_TYPE && rhs.type_ == NIL_TYPE);
  }
}


template <class T>

  inline bool
  Ref<Object>::operator == ( const Ref<T>& rhs ) const noexcept

{
  return operator == ( rhs.get() );
}


template <class T>

  inline bool
  Ref<Object>::operator == ( const T* rhs ) const noexcept

{
  if ( type_ == NIL_TYPE || type_ == OBJECT_TYPE )
  {
    return (data_.object == rhs);
  }
  else
  {
    return false;
  }
}


inline bool
  Ref<Object>::operator == ( const Nil& rhs ) const noexcept
{
  return (type_ == NIL_TYPE);
}


inline bool
  Ref<Object>::operator == ( std::nullptr_t rhs ) const noexcept
{
  return (type_ == NIL_TYPE);
}


//-----------------------------------------------------------------------
//   access operators
//-----------------------------------------------------------------------


inline Object* Ref<Object>::operator -> () const
{
  JEM_ASSERT ( checkAddr_() );

  if ( type_ != OBJECT_TYPE )
  {
    toObject_ ();
  }

  return data_.object;
}


inline Object& Ref<Object>::operator  * () const
{
  JEM_ASSERT ( checkAddr_() );

  if ( type_ != OBJECT_TYPE )
  {
    toObject_ ();
  }

  return *data_.object;
}


//-----------------------------------------------------------------------
//   bool operators
//-----------------------------------------------------------------------


constexpr inline Ref<Object>::operator bool () const noexcept
{
  return (type_ != NIL_TYPE);
}


constexpr inline bool Ref<Object>::operator ! () const noexcept
{
  return (type_ == NIL_TYPE);
}


//-----------------------------------------------------------------------
//   get & getBase
//-----------------------------------------------------------------------


inline Object* Ref<Object>::get () const
{
  if ( type_ != OBJECT_TYPE && type_ != NIL_TYPE )
  {
    toObject_ ();
  }

  return data_.object;
}


inline Collectable* Ref<Object>::getBase () const
{
  return get ();
}


//-----------------------------------------------------------------------
//   release
//-----------------------------------------------------------------------


inline void Ref<Object>::release () noexcept
{
  type_        = NIL_TYPE;
  data_.object = nullptr;
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Ref<Object>::TypeID Ref<Object>::getType () const noexcept
{
  return type_;
}


//-----------------------------------------------------------------------
//   getBool
//-----------------------------------------------------------------------


inline bool Ref<Object>::getBool () const
{
  JEM_ASSERT2 ( getType() == BOOL_TYPE,
                "invalid type (expected a boolean)" );

  return (data_.ivalue != 0);
}


//-----------------------------------------------------------------------
//   getChar
//-----------------------------------------------------------------------


inline char Ref<Object>::getChar () const
{
  JEM_ASSERT2 ( getType() == CHAR_TYPE,
                "invalid type (expected a character)" );

  return (char) data_.ivalue;
}


//-----------------------------------------------------------------------
//   getInt
//-----------------------------------------------------------------------


inline lint Ref<Object>::getInt () const
{
  JEM_ASSERT2 ( getType() == INT_TYPE,
                "invalid type (expected an integer)" );

  return data_.ivalue;
}


//-----------------------------------------------------------------------
//   getFloat
//-----------------------------------------------------------------------


inline double Ref<Object>::getFloat () const
{
  JEM_ASSERT2 ( getType() == FLOAT_TYPE,
                "invalid type (expected a floating point number)" );

  return data_.xvalue;
}


//-----------------------------------------------------------------------
//   getString
//-----------------------------------------------------------------------


inline String Ref<Object>::getString () const
{
  JEM_ASSERT2 ( getType() == STRING_TYPE,
                "invalid type (expected a string)" );

  return TinyString ( data_.strptr );
}


//-----------------------------------------------------------------------
//   getObject
//-----------------------------------------------------------------------


inline Object& Ref<Object>::getObject () const
{
  JEM_ASSERT2 ( getType() == OBJECT_TYPE,
                "invalid type (expected an Object)" );

  return *data_.object;
}


//-----------------------------------------------------------------------
//   checkAddr_
//-----------------------------------------------------------------------


inline bool Ref<Object>::checkAddr_ () const
{
  if ( type_ == NIL_TYPE )
  {
    nullError_ ();
  }

  return true;
}


//=======================================================================
//   class DynRefCaster<T,Object>
//=======================================================================


template <class T>

  class DynRefCaster<T,Object>

{
 public:

  static inline T*        dynamicCast

    ( const Ref<Object>&    rhs );


 private:

  static inline T*        dynamicCast_

    ( const Ref<Object>&    rhs,
      void*                 target );

  static inline T*        dynamicCast_

    ( const Ref<Object>&    rhs,
      Object*               target );

  static inline T*        dynamicCast2_

    ( Object&               rhs,
      void*                 target );

  static inline T*        dynamicCast2_

    ( Object&               rhs,
      T*                    target );

};


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T>

  inline T* DynRefCaster<T,Object>::dynamicCast

  ( const Ref<Object>& rhs )

{
  return dynamicCast_ ( rhs, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   dynamicCast_
//-----------------------------------------------------------------------


template <class T>

  inline T* DynRefCaster<T,Object>::dynamicCast_

  ( const Ref<Object>&  rhs,
    void*               target )

{
  return dynamic_cast<T*> ( rhs.get() );
}


template <class T>

  inline T* DynRefCaster<T,Object>::dynamicCast_

  ( const Ref<Object>&  rhs,
    Object*             target )

{
  if ( rhs.getType() == Ref<Object>::OBJECT_TYPE )
  {
    return dynamicCast2_ ( rhs.getObject(),
                           (typename T::DeclaringClass*) nullptr );
  }
  else
  {
    return (T*) nullptr;
  }
}


//-----------------------------------------------------------------------
//   dynamicCast2_
//-----------------------------------------------------------------------


template <class T>

  inline T* DynRefCaster<T,Object>::dynamicCast2_

  ( Object&  rhs,
    void*    target )

{
  return dynamic_cast<T*> ( &rhs );
}


template <class T>

  inline T* DynRefCaster<T,Object>::dynamicCast2_

  ( Object&  rhs,
    T*       target )

{
  return static_cast<T*>

    ( Ref<Object>::dynamicCast( rhs, T::getType() ) );
}


//=======================================================================
//   class DynRefCaster<Object,Object>
//=======================================================================


template <>

  class DynRefCaster<Object,Object>

{
 public:

  static Object*          dynamicCast

    ( const Ref<Object>&    rhs )

  {
    return rhs.get ();
  }

};


JEM_END_PACKAGE_BASE

#endif
