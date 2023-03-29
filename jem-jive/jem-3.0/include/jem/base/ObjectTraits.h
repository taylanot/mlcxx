
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

#ifndef JEM_BASE_OBJECTTRAITS_H
#define JEM_BASE_OBJECTTRAITS_H

#include <jem/base/BoxObject.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ObjectTraitsUtils
//-----------------------------------------------------------------------


class ObjectTraitsUtils
{
 public:

  typedef Ref<Object>       ObjRef;


  template <class T>

    static T*               ptrCast

    ( const ObjRef&           obj );

  template <class T>

    static T&               refCast

    ( const ObjRef&           obj );

  static void               typeError

    ( const ObjRef&           obj,
      Class*                  target );

  static void               rangeError

    ( lint                    value,
      lint                    lbound,
      lint                    ubound );

  static void               rangeError

    ( double                  value,
      double                  ubound );

};


//-----------------------------------------------------------------------
//   class ObjectTraits (primary template)
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits : private ObjectTraitsUtils

{
 public:

  typedef BoxObject<T>      BoxType;


  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( const T&                value );

  static inline const T&    toValue

    ( const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static inline Class*      getType_

    ( Object*                 type );

  static inline Class*      getType_

    ( void*                   type );

  static inline ObjRef      toObject_

    ( const T&                value,
      Object*                 type );

  static inline ObjRef      toObject_

    ( const T&                value,
      void*                   type );

  static inline const T&    toValue_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline const T&    toValue_

    ( const ObjRef&           obj,
      void*                   type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      void*                   type );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<void>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<void> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<bool>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<bool> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static inline ObjRef      toObject

    ( bool                    value );

  static inline bool        toValue

    ( const ObjRef&           obj );

  static inline bool        toValue

    ( bool&                   value,
      const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static bool               toValue_

    ( const ObjRef&           obj );

  static bool               toValue_

    ( bool&                   value,
      const ObjRef&           obj );

  static bool               isInstance_

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<char>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<char> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static inline ObjRef      toObject

    ( char                    value );

  static inline char        toValue

    ( const ObjRef&           obj );

  static inline bool        toValue

    ( char&                   value,
      const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static char               toValue_

    ( const ObjRef&           obj );

  static bool               toValue_

    ( char&                   value,
      const ObjRef&           obj );

  static bool               isInstance_

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<byte>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<byte> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( byte                    value );

  static byte               toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( byte&                   value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<short>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<short> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( short                   value );

  static short              toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( short&                  value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<int>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<int> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( int                     value );

  static int                toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( int&                    value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<long>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<long> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( long                    value );

  static long               toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( long&                   value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<lint>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<lint> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static inline ObjRef      toObject

    ( lint                    value );

  static inline lint        toValue

    ( const ObjRef&           obj );

  static inline bool        toValue

    ( lint&                   value,
      const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static lint               toValue_

    ( const ObjRef&           obj );

  static bool               toValue_

    ( lint&                   value,
      const ObjRef&           obj );

  static bool               isInstance_

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<idx_t>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<idx_t> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( idx_t                   value );

  static idx_t              toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( idx_t&                  value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<float>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<float> : private ObjectTraitsUtils

{
 public:

  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( float                   value );

  static float              toValue

    ( const ObjRef&           obj );

  static bool               toValue

    ( float&                  value,
      const ObjRef&           obj );

  static bool               isInstance

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<double>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<double> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static inline ObjRef      toObject

    ( double                  value );

  static inline double      toValue

    ( const ObjRef&           obj );

  static inline bool        toValue

    ( double&                 value,
      const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static double             toValue_

    ( const ObjRef&           obj );

  static bool               toValue_

    ( double&                 value,
      const ObjRef&           obj );

  static bool               isInstance_

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<String>
//-----------------------------------------------------------------------


template <>

  class ObjectTraits<String> : private ObjectTraitsUtils

{
 public:

  static Class*             getType ();

  static inline ObjRef      toObject

    ( const String&           value );

  static inline String      toValue

    ( const ObjRef&           obj );

  static inline bool        toValue

    ( String&                 value,
      const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static String             toValue_

    ( const ObjRef&           obj );

  static bool               toValue_

    ( String&                 value,
      const ObjRef&           obj );

  static bool               isInstance_

    ( const ObjRef&           obj );

};


//-----------------------------------------------------------------------
//   class ObjectTraits< Ref<T> >
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits< Ref<T> > : private ObjectTraitsUtils

{
 public:

  typedef
    BoxObject< Ref<T> >     BoxType;


  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( const Ref<T>&           value );

  static inline Ref<T>      toValue

    ( const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static inline Class*      getType_

    ( Object*                 type );

  static inline Class*      getType_

    ( void*                   type );

  static inline ObjRef      toObject_

    ( const Ref<T>&           value,
      Object*                 type );

  static inline ObjRef      toObject_

    ( const Ref<T>&           value,
      void*                   type );

  static inline Ref<T>      toValue_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline Ref<T>      toValue_

    ( const ObjRef&           obj,
      void*                   type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      void*                   type );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<T&>
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits<T&> : private ObjectTraitsUtils

{
 public:

  typedef BoxObject<T>      BoxType;


  static inline Class*      getType ();

  static inline T&          toValue

    ( const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static inline Class*      getType_

    ( Object*                 type );

  static inline Class*      getType_

    ( void*                   type );

  static inline T&          toValue_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline T&          toValue_

    ( const ObjRef&           obj,
      void*                   type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      void*                   type );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<const T&>
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits<const T&> : public ObjectTraits<T>

{};


//-----------------------------------------------------------------------
//   class ObjectTraits<T*>
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits<T*> : private ObjectTraitsUtils

{
 public:

  typedef BoxObject<T>      BoxType;


  static inline Class*      getType ();

  static inline ObjRef      toObject

    ( const T*                value );

  static inline T*          toValue

    ( const ObjRef&           obj );

  static inline bool        isInstance

    ( const ObjRef&           obj );


 private:

  static inline Class*      getType_

    ( Object*                 type );

  static inline Class*      getType_

    ( void*                   type );

  static inline ObjRef      toObject_

    ( const T*                value,
      Object*                 type );

  static inline ObjRef      toObject_

    ( const T*                value,
      void*                   type );

  static inline T*          toValue_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline T*          toValue_

    ( const ObjRef&           obj,
      void*                   type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      Object*                 type );

  static inline bool        isInstance_

    ( const ObjRef&           obj,
      void*                   type );

};


//-----------------------------------------------------------------------
//   class ObjectTraits<const T*>
//-----------------------------------------------------------------------


template <class T>

  class ObjectTraits<const T*> : public ObjectTraits<T*>

{};







//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ObjectTraitsUtils
//=======================================================================


template <class T>

  T* ObjectTraitsUtils::ptrCast ( const ObjRef& rhs )

{
  T*  tmp = DynRefCaster<T,Object>::dynamicCast ( rhs );

  if ( ! tmp && rhs )
  {
    typeError ( rhs, T::getType() );
  }

  return tmp;
}


template <class T>

  T& ObjectTraitsUtils::refCast ( const ObjRef& rhs )

{
  T*  tmp = DynRefCaster<T,Object>::dynamicCast ( rhs );

  if ( ! tmp )
  {
    typeError ( rhs, T::getType() );
  }

  return *tmp;
}


//=======================================================================
//   class ObjectTraits<T>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T>::getType ()

{
  return getType_ ( (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits<T>::toObject ( const T& value )

{
  return toObject_ ( value, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


template <class T>

  inline const T& ObjectTraits<T>::toValue ( const ObjRef& obj )

{
  return toValue_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T>::isInstance ( const ObjRef& obj )

{
  return isInstance_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   getType_
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T>::getType_ ( Object* type )

{
  return T::getType ();
}


template <class T>

  inline Class* ObjectTraits<T>::getType_ ( void* type )

{
  return BoxType::getType ();
}


//-----------------------------------------------------------------------
//   toObject_
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits<T>::toObject_

  ( const T&  value,
    Object*   type )

{
  return ObjRef ( const_cast<T*>( &value ) );
}


template <class T>

  inline Ref<Object> ObjectTraits<T>::toObject_

  ( const T&  value,
    void*     type )

{
  return newInstance<BoxType> ( value );
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


template <class T>

  inline const T& ObjectTraits<T>::toValue_

  ( const ObjRef&  obj,
    Object*        type )

{
  return refCast<T> ( obj );
}


template <class T>

  inline const T& ObjectTraits<T>::toValue_

  ( const ObjRef&  obj,
    void*          type )

{
  return refCast<BoxType>( obj ).value;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T>::isInstance_

  ( const ObjRef&  obj,
    Object*        type )

{
  return (DynRefCaster<T,Object>::dynamicCast( obj ) != 0);
}


template <class T>

  inline bool ObjectTraits<T>::isInstance_

  ( const ObjRef&  obj,
    void*          type )

{
  return (DynRefCaster<BoxType,Object>::dynamicCast( obj ) != 0);
}


//=======================================================================
//   class ObjectTraits<bool>
//=======================================================================

//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<bool>::toObject ( bool value )
{
  return ObjRef ( value, RAW_REF );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


inline bool ObjectTraits<bool>::toValue ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::BOOL_TYPE )
  {
    return obj.getBool ();
  }
  else
  {
    return toValue_ ( obj );
  }
}


inline bool ObjectTraits<bool>::toValue

  ( bool&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::BOOL_TYPE )
  {
    value = obj.getBool ();

    return true;
  }
  else
  {
    return toValue_ ( value, obj );
  }
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


inline bool ObjectTraits<bool>::isInstance ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::BOOL_TYPE )
  {
    return true;
  }
  else
  {
    return isInstance_ ( obj );
  }
}


//=======================================================================
//   class ObjectTraits<char>
//=======================================================================

//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<char>::toObject ( char value )
{
  return ObjRef ( value, RAW_REF );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


inline char ObjectTraits<char>::toValue ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    return obj.getChar ();
  }
  else
  {
    return toValue_ ( obj );
  }
}


inline bool ObjectTraits<char>::toValue

  ( char&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    value = obj.getChar ();

    return true;
  }
  else
  {
    return toValue_ ( value, obj );
  }
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


inline bool ObjectTraits<char>::isInstance ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::CHAR_TYPE )
  {
    return true;
  }
  else
  {
    return isInstance_ ( obj );
  }
}


//=======================================================================
//   clas ObjectTraits<byte>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<byte>::getType ()
{
  return ObjectTraits<lint>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<byte>::toObject ( byte value )
{
  return ObjRef ( (lint) value, RAW_REF );
}


//=======================================================================
//   class ObjectTrait<short>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<short>::getType ()
{
  return ObjectTraits<lint>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<short>::toObject ( short value )
{
  return ObjRef ( (lint) value, RAW_REF );
}


//=======================================================================
//   class ObjectTraits<int>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<int>::getType ()
{
  return ObjectTraits<lint>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<int>::toObject ( int value )
{
  return ObjRef ( (lint) value, RAW_REF );
}


//=======================================================================
//   class ObjectTraits<long>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<long>::getType ()
{
  return ObjectTraits<lint>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<long>::toObject ( long value )
{
  return ObjRef ( (lint) value, RAW_REF );
}


//=======================================================================
//   class ObjectTraits<lint>
//=======================================================================

//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<lint>::toObject ( lint value )
{
  return ObjRef ( value, RAW_REF );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


inline lint ObjectTraits<lint>::toValue ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::INT_TYPE )
  {
    return obj.getInt ();
  }
  else
  {
    return toValue_ ( obj );
  }
}


inline bool ObjectTraits<lint>::toValue

  ( lint&          value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::INT_TYPE )
  {
    value = obj.getInt ();

    return true;
  }
  else
  {
    return toValue_ ( value, obj );
  }
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


inline bool ObjectTraits<lint>::isInstance ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::INT_TYPE )
  {
    return true;
  }
  else
  {
    return isInstance_ ( obj );
  }
}


//=======================================================================
//   class ObjectTraits<idx_t>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<idx_t>::getType ()
{
  return ObjectTraits<lint>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<idx_t>::toObject ( idx_t value )
{
  return ObjRef ( (lint) value, RAW_REF );
}


//=======================================================================
//   class ObjectTraits<float>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* ObjectTraits<float>::getType ()
{
  return ObjectTraits<double>::getType ();
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<float>::toObject ( float value )
{
  return ObjRef ( (double) value, RAW_REF );
}


//=======================================================================
//   class ObjectTraits<double>
//=======================================================================

//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object> ObjectTraits<double>::toObject ( double value )
{
  return ObjRef ( value, RAW_REF );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


inline double ObjectTraits<double>::toValue ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::FLOAT_TYPE )
  {
    return obj.getFloat ();
  }
  else
  {
    return toValue_ ( obj );
  }
}


inline bool ObjectTraits<double>::toValue

  ( double&        value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::FLOAT_TYPE )
  {
    value = obj.getFloat ();

    return true;
  }
  else
  {
    return toValue_ ( value, obj );
  }
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


inline bool ObjectTraits<double>::isInstance ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::FLOAT_TYPE )
  {
    return true;
  }
  else
  {
    return isInstance_ ( obj );
  }
}


//=======================================================================
//   class ObjectTraits<String>
//=======================================================================

//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


inline Ref<Object>
  ObjectTraits<String>::toObject ( const String& value )

{
  return ObjRef ( value, RAW_REF );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


inline String ObjectTraits<String>::toValue ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::STRING_TYPE )
  {
    return obj.getString ();
  }
  else
  {
    return toValue_ ( obj );
  }
}


inline bool ObjectTraits<String>::toValue

  ( String&        value,
    const ObjRef&  obj )

{
  if ( obj.getType() == ObjRef::STRING_TYPE )
  {
    value = obj.getString ();

    return true;
  }
  else
  {
    return toValue_ ( value, obj );
  }
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


inline bool ObjectTraits<String>::isInstance ( const ObjRef& obj )
{
  if ( obj.getType() == ObjRef::STRING_TYPE )
  {
    return true;
  }
  else
  {
    return isInstance_ ( obj );
  }
}


//=======================================================================
//   class ObjectTraits< Ref<T> >
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits< Ref<T> >::getType ()

{
  return getType_ ( (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits< Ref<T> >::toObject

  ( const Ref<T>&  value )

{
  return toObject_ ( value, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


template <class T>

  inline Ref<T> ObjectTraits< Ref<T> >::toValue

  ( const ObjRef&  obj )

{
  return toValue_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits< Ref<T> >::isInstance

  ( const ObjRef&  obj )

{
  return isInstance_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   getType_
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits< Ref<T> >::getType_

  ( Object*  type )

{
  return T::getType ();
}


template <class T>

  inline Class* ObjectTraits< Ref<T> >::getType_

  ( void*  type )

{
  return BoxType::getType ();
}


//-----------------------------------------------------------------------
//   toObject_
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits< Ref<T> >::toObject_

  ( const Ref<T>&  value,
    Object*        type )

{
  return value;
}


template <class T>

  inline Ref<Object> ObjectTraits< Ref<T> >::toObject_

  ( const Ref<T>&  value,
    void*          type )

{
  return newInstance<BoxType> ( value );
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


template <class T>

  inline Ref<T> ObjectTraits< Ref<T> >::toValue_

  ( const ObjRef&  obj,
    Object*        type )

{
  return Ref<T> ( ptrCast<T>( obj ) );
}


template <class T>

  inline Ref<T> ObjectTraits< Ref<T> >::toValue_

  ( const ObjRef&  obj,
    void*          type )

{
  return refCast<BoxType>( obj ).value;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits< Ref<T> >::isInstance_

  ( const ObjRef&  obj,
    Object*        type )

{
  return (bool) DynRefCaster<T,Object>::dynamicCast ( obj );
}


template <class T>

  inline bool ObjectTraits< Ref<T> >::isInstance_

  ( const ObjRef&  obj,
    void*          type )

{
  return (bool) DynRefCaster<BoxType,Object>::dynamicCast ( obj );
}


//=======================================================================
//   class ObjectTraits<T&>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T&>::getType ()

{
  return getType_ ( (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


template <class T>

  inline T& ObjectTraits<T&>::toValue ( const ObjRef& obj )

{
  return toValue_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T&>::isInstance ( const ObjRef& obj )

{
  return isInstance_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   getType_
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T&>::getType_ ( Object* type )

{
  return T::getType ();
}


template <class T>

  inline Class* ObjectTraits<T&>::getType_ ( void* type )

{
  return BoxType::getType ();
}


//-----------------------------------------------------------------------
//   toValue_
//-----------------------------------------------------------------------


template <class T>

  inline T& ObjectTraits<T&>::toValue_

  ( const ObjRef&  obj,
    Object*        type )

{
  return refCast<T> ( obj );
}


template <class T>

  inline T& ObjectTraits<T&>::toValue_

  ( const ObjRef&  obj,
    void*          type )

{
  return refCast<BoxType>( obj ).value;
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T&>::isInstance_

  ( const ObjRef&  obj,
    Object*        type )

{
  return (bool) DynRefCaster<T,Object>::dynamicCast ( obj );
}


template <class T>

  inline bool ObjectTraits<T&>::isInstance_

  ( const ObjRef&  obj,
    void*          type )

{
  return (bool) DynRefCaster<BoxType,Object>::dynamicCast ( obj );
}


//=======================================================================
//   class ObjectTraits<T*>
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T*>::getType ()

{
  return getType_ ( (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toObject
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits<T*>::toObject ( const T* value )

{
  return toObject_ ( value, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


template <class T>

  inline T* ObjectTraits<T*>::toValue ( const ObjRef& obj )

{
  return toValue_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T*>::isInstance ( const ObjRef& obj )

{
  return isInstance_ ( obj, (T*) nullptr );
}


//-----------------------------------------------------------------------
//   getType_
//-----------------------------------------------------------------------


template <class T>

  inline Class* ObjectTraits<T*>::getType_ ( Object* type )

{
  return T::getType ();
}


template <class T>

  inline Class* ObjectTraits<T*>::getType_ ( void* type )

{
  return BoxType::getType ();
}


//-----------------------------------------------------------------------
//   toObject_
//-----------------------------------------------------------------------


template <class T>

  inline Ref<Object> ObjectTraits<T*>::toObject_

  ( const T*  value,
    Object*   type )

{
  return ObjRef ( const_cast<T*>( value ) );
}


template <class T>

  inline Ref<Object> ObjectTraits<T*>::toObject_

  ( const T*  value,
    void*     type )

{
  if ( value )
  {
    return newInstance<BoxType> ( *value );
  }
  else
  {
    return nullptr;
  }
}


//-----------------------------------------------------------------------
//   toValue
//-----------------------------------------------------------------------


template <class T>

  inline T* ObjectTraits<T*>::toValue_

  ( const ObjRef&  obj,
    Object*        type )

{
  return ptrCast<T> ( obj );
}


template <class T>

  inline T* ObjectTraits<T*>::toValue_

  ( const ObjRef&  obj,
    void*          type )

{
  if ( ! obj )
  {
    return (T*) nullptr;
  }
  else
  {
    return & refCast<BoxType>( obj ).value;
  }
}


//-----------------------------------------------------------------------
//   isInstance_
//-----------------------------------------------------------------------


template <class T>

  inline bool ObjectTraits<T*>::isInstance_

  ( const ObjRef&  obj,
    Object*        type )

{
  return (bool) DynRefCaster<T,Object>::dynamicCast ( obj );
}


template <class T>

  inline bool ObjectTraits<T*>::isInstance_

  ( const ObjRef&  obj,
    void*          type )

{
  return (bool) DynRefCaster<BoxType,Object>::dynamicCast ( obj );
}


JEM_END_PACKAGE_BASE

#endif
