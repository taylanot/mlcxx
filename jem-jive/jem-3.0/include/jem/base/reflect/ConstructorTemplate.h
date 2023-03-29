
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

#ifndef JEM_BASE_REFLECT_CONSTRUCTORTEMPLATE_H
#define JEM_BASE_REFLECT_CONSTRUCTORTEMPLATE_H

#include <jem/base/ObjectTraits.h>
#include <jem/base/reflect/Constructor.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ConstructorTemplate
//-----------------------------------------------------------------------


template <class T, class A = Nil, class B = Nil, class C = Nil>

  class ConstructorTemplate : public Constructor

{
 public:

  typedef Constructor   Super;

  static const int      ARG_COUNT = 3;


  inline                ConstructorTemplate ();

  virtual idx_t         argCount            () const override;

  virtual void          getArgTypes

    ( Class**             types )              const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args )               const override;


 protected:

  virtual              ~ConstructorTemplate ();


 private:

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      Object*             dummy )              const;

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      void*               dummy )              const;

};


//-----------------------------------------------------------------------
//   class ConstructorTemplate<T,A,B,Nil>
//-----------------------------------------------------------------------


template <class T, class A, class B>

  class ConstructorTemplate<T,A,B,Nil> : public Constructor

{
 public:

  typedef Constructor   Super;

  static const int      ARG_COUNT = 2;


  inline                ConstructorTemplate ();

  virtual idx_t         argCount            () const override;

  virtual void          getArgTypes

    ( Class**             types )              const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args )               const override;


 protected:

  virtual              ~ConstructorTemplate ();


 private:

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      Object*             dummy )              const;

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      void*               dummy )              const;

};


//-----------------------------------------------------------------------
//   class ConstructorTemplate<T,A,Nil,Nil>
//-----------------------------------------------------------------------


template <class T, class A>

  class ConstructorTemplate<T,A,Nil,Nil> : public Constructor

{
 public:

  typedef Constructor   Super;


  static const int      ARG_COUNT = 1;


  inline                ConstructorTemplate ();

  virtual idx_t         argCount            () const override;

  virtual void          getArgTypes

    ( Class**             types )              const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args )               const override;


 protected:

  virtual              ~ConstructorTemplate ();


 private:

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      Object*             dummy )              const;

  inline Ref<Object>    call_

    ( const Ref<Object>*  args,
      void*               dummy )              const;

};


//-----------------------------------------------------------------------
//   class ConstructorTemplate<T,Nil,Nil,Nil>
//-----------------------------------------------------------------------


template <class T>

  class ConstructorTemplate<T,Nil,Nil,Nil> : public Constructor

{
 public:

  typedef Constructor   Super;

  static const int      ARG_COUNT = 0;


  inline                ConstructorTemplate ();

  virtual idx_t         argCount            () const override;

  virtual void          getArgTypes

    ( Class**             types )              const override;

  virtual Ref<Object>   call

    ( const Ref<Object>*  args )               const override;


 protected:

  virtual              ~ConstructorTemplate ();


 private:

  inline Ref<Object>    call_

    ( Object*             dummy )              const;

  inline Ref<Object>    call_

    ( void*               dummy )              const;

};






//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ConstructorTemplate
//=======================================================================


template <class T, class A, class B, class C>

  inline ConstructorTemplate<T,A,B,C>::ConstructorTemplate ()

{}


template <class T, class A, class B, class C>

  ConstructorTemplate<T,A,B,C>::~ConstructorTemplate ()

{}


template <class T, class A, class B, class C>

  idx_t ConstructorTemplate<T,A,B,C>::argCount () const

{
  return (idx_t) ARG_COUNT;
}


template <class T, class A, class B, class C>

  void ConstructorTemplate<T,A,B,C>::getArgTypes

  ( Class** types ) const

{
  types[0] = ObjectTraits<A>::getType ();
  types[1] = ObjectTraits<B>::getType ();
  types[2] = ObjectTraits<C>::getType ();
}


template <class T, class A, class B, class C>

  Ref<Object> ConstructorTemplate<T,A,B,C>::call

  ( const Ref<Object>* args ) const

{
  return call_ ( args, (T*) nullptr );
}


template <class T, class A, class B, class C>

  inline Ref<Object> ConstructorTemplate<T,A,B,C>::call_

  ( const Ref<Object>* args, Object* ) const

{
  return newInstance<T>(
    ObjectTraits<A>::toValue ( args[0] ),
    ObjectTraits<B>::toValue ( args[1] ),
    ObjectTraits<C>::toValue ( args[2] )
  );
}


template <class T, class A, class B, class C>

  inline Ref<Object> ConstructorTemplate<T,A,B,C>::call_

  ( const Ref<Object>* args, void* ) const

{
  return ObjectTraits<T>::toObject (
    T ( ObjectTraits<A>::toValue ( args[0] ),
        ObjectTraits<B>::toValue ( args[1] ),
        ObjectTraits<C>::toValue ( args[2] ) )
  );
}


//=======================================================================
//   class ConstructorTemplate<T,A,B,Nil>
//=======================================================================


template <class T, class A, class B>

  inline ConstructorTemplate<T,A,B,Nil>::ConstructorTemplate ()

{}


template <class T, class A, class B>

  ConstructorTemplate<T,A,B,Nil>::~ConstructorTemplate ()

{}


template <class T, class A, class B>

  idx_t ConstructorTemplate<T,A,B,Nil>::argCount () const

{
  return (idx_t) ARG_COUNT;
}


template <class T, class A, class B>

  void ConstructorTemplate<T,A,B,Nil>::getArgTypes

  ( Class** types ) const

{
  types[0] = ObjectTraits<A>::getType();
  types[1] = ObjectTraits<B>::getType();
}


template <class T, class A, class B>

  Ref<Object> ConstructorTemplate<T,A,B,Nil>::call

  ( const Ref<Object>* args ) const

{
  return call_( args, (T*) nullptr );
}


template <class T, class A, class B>

  inline Ref<Object> ConstructorTemplate<T,A,B,Nil>::call_

  ( const Ref<Object>* args, Object* ) const

{
  return newInstance<T>(
    ObjectTraits<A>::toValue ( args[0] ),
    ObjectTraits<B>::toValue ( args[1] )
  );
}


template <class T, class A, class B>

  inline Ref<Object> ConstructorTemplate<T,A,B,Nil>::call_

  ( const Ref<Object>* args, void* ) const

{
  return ObjectTraits<T>::toObject (
    T ( ObjectTraits<A>::toValue ( args[0] ),
        ObjectTraits<B>::toValue ( args[1] ) )
  );
}


//=======================================================================
//   class ConstructorTemplate<T,A,Nil,Nil>
//=======================================================================


template <class T, class A>

  inline
  ConstructorTemplate<T,A,Nil,Nil>::ConstructorTemplate ()

{}


template <class T, class A>

  ConstructorTemplate<T,A,Nil,Nil>::~ConstructorTemplate ()

{}


template <class T, class A>

  idx_t ConstructorTemplate<T,A,Nil,Nil>::argCount () const

{
  return (idx_t) ARG_COUNT;
}


template <class T, class A>

  void ConstructorTemplate<T,A,Nil,Nil>::getArgTypes

  ( Class** types ) const

{
  types[0] = ObjectTraits<A>::getType ();
}


template <class T, class A>

  Ref<Object> ConstructorTemplate<T,A,Nil,Nil>::call

  ( const Ref<Object>* args ) const

{
  return call_ ( args, (T*) nullptr );
}


template <class T, class A>

  inline Ref<Object> ConstructorTemplate<T,A,Nil,Nil>::call_

  ( const Ref<Object>* args, Object* ) const

{
  return newInstance<T>( ObjectTraits<A>::toValue( args[0] ) );
}


template <class T, class A>

  inline Ref<Object> ConstructorTemplate<T,A,Nil,Nil>::call_

  ( const Ref<Object>* args, void* ) const

{
  return ObjectTraits<T>::toObject (
    T( ObjectTraits<A>::toValue( args[0] ) )
  );
}


//=======================================================================
//   class ConstructorTemplate<T,Nil,Nil,Nil>
//=======================================================================


template <class T>

  inline
  ConstructorTemplate<T,Nil,Nil,Nil>::ConstructorTemplate ()

{}


template <class T>

  ConstructorTemplate<T,Nil,Nil,Nil>::~ConstructorTemplate ()

{}


template <class T>

  idx_t ConstructorTemplate<T,Nil,Nil,Nil>::argCount () const

{
  return (idx_t) ARG_COUNT;
}


template <class T>

  void ConstructorTemplate<T,Nil,Nil,Nil>::getArgTypes

  ( Class** ) const

{}


template <class T>

  Ref<Object> ConstructorTemplate<T,Nil,Nil,Nil>::call

  ( const Ref<Object>* ) const

{
  return call_ ( (T*) nullptr );
}


template <class T>

  inline Ref<Object> ConstructorTemplate<T,Nil,Nil,Nil>::call_

  ( Object* ) const

{
  return newInstance<T> ();
}


template <class T>

  inline Ref<Object> ConstructorTemplate<T,Nil,Nil,Nil>::call_

  ( void* ) const

{
  return ObjectTraits<T>::toObject ( T() );
}


JEM_END_PACKAGE_BASE

#endif
