
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

#ifndef JEM_BASE_BOXOBJECT_H
#define JEM_BASE_BOXOBJECT_H

#include <jem/base/Clonable.h>
#include <jem/base/TypeTraits.h>
#include <jem/base/ClassTemplate.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/io/Serializable.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class BoxObject ( primary template )
//-----------------------------------------------------------------------


template < class T,
           bool  IsSerializable = TypeTraits<T>::IS_SERIALIZABLE >

  class BoxObject

{};


//-----------------------------------------------------------------------
//   class BoxObject<T,false>
//-----------------------------------------------------------------------


template <class T>

  class BoxObject<T,false> : public Object,
                             public Clonable
{
 public:

  typedef Object              Super;
  typedef BoxObject<T,false>  Self;
  typedef BoxObject<T,false>  DeclaringClass;


  explicit inline             BoxObject

    ( const T&                  value );

  static  Class*              makeClass   ();
  static  Class*              getType     ();
  virtual Class*              getClass    () const override;
  virtual Ref<Object>         clone       () const override;


 public:

  T                           value;


 protected:

  virtual                    ~BoxObject   ();

};


//-----------------------------------------------------------------------
//   class BoxObject<T,true>
//-----------------------------------------------------------------------


template <class T>

  class BoxObject<T,true> : public Object,
                            public Clonable,
                            public io::Serializable
{
 public:

  typedef Object              Super;
  typedef BoxObject<T,true>   Self;
  typedef BoxObject<T,true>   DeclaringClass;


  inline                      BoxObject   ();

  explicit inline             BoxObject

    ( const T&                  value );

  static  Class*              makeClass   ();
  static  Class*              getType     ();
  virtual Class*              getClass    () const override;

  virtual void                readFrom

    ( io::ObjectInput&          in  )              override;

  virtual void                writeTo

    ( io::ObjectOutput&         out )        const override;

  virtual Ref<Object>         clone       () const override;


 public:

  T                           value;


 protected:

  virtual                    ~BoxObject   ();

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class BoxObject<T,false>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  inline BoxObject<T,false>::BoxObject ( const T& val ) :

  value ( val )

{}


template <class T>

  BoxObject<T,false>::~BoxObject ()

{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,false>::makeClass ()

{
  ClassTemplate<Self>*  myClass =

    new ClassTemplate<Self> ( Super::getType(),
                              TypeTraits<T>::typeName() );

  myClass->setWhatis ( TypeTraits<T>::whatis() );

  return myClass;
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,false>::getType ()

{
  return ClassTemplate<Self>::getInstance ();
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,false>::getClass () const

{
  return getType ();
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


template <class T>

  Ref<Object> BoxObject<T,false>::clone () const

{
  return newInstance<Self> ( value );
}


//=======================================================================
//   class BoxObject<T,true>
//=======================================================================

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  inline BoxObject<T,true>::BoxObject ()

{}


template <class T>

  inline BoxObject<T,true>::BoxObject ( const T& val ) :

    value ( val )

{}


template <class T>

  BoxObject<T,true>::~BoxObject ()

{}


//-----------------------------------------------------------------------
//   makeClass
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,true>::makeClass ()

{
  ClassTemplate<Self>*  myClass =

    new ClassTemplate<Self> ( Super::getType (),
                              TypeTraits<T>::typeName () );

  myClass->setWhatis      ( TypeTraits<T>::whatis() );
  myClass->addDefaultCtor ();

  return myClass;
}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,true>::getType ()

{
  return ClassTemplate<Self>::getInstance ();
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


template <class T>

  Class* BoxObject<T,true>::getClass () const

{
  return getType ();
}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


template <class T>

  void BoxObject<T,true>::readFrom ( io::ObjectInput& in )

{
  decode ( in, value );
}


template <class T>

  void BoxObject<T,true>::writeTo ( io::ObjectOutput& out ) const

{
  encode ( out, value );
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


template <class T>

  Ref<Object> BoxObject<T,true>::clone () const

{
  return newInstance<Self> ( TypeTraits<T>::clone( value ) );
}


JEM_END_PACKAGE_BASE

#endif
