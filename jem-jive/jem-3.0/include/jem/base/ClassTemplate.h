
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

#ifndef JEM_BASE_CLASSTEMPLATE_H
#define JEM_BASE_CLASSTEMPLATE_H

#include <typeinfo>
#include <jem/base/Class.h>
#include <jem/base/Once.h>
#include <jem/base/reflect/DefaultConstructor.h>


JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   class ClassTemplate
//-----------------------------------------------------------------------


template <class T>

  class ClassTemplate : public Class

{
 public:

  inline                    ClassTemplate   ();

  explicit inline           ClassTemplate

    ( const String&           name );

  inline                    ClassTemplate

    ( Class*                  super,
      const String&           name );

  virtual bool              equalType

    ( const Ref<Object>&      obj )            const override;

  inline void               addDefaultCtor  ();
  static inline Class*      getInstance     ();

  static inline T*          dynamicCast

    ( Object&                 rhs );


 protected:

  virtual                  ~ClassTemplate   ();


 private:

  static void               init_           ();
  static inline Class*      getSuper_       ();


 private:

  static Once               once_;
  static Class*             instance_;

};


//-----------------------------------------------------------------------
//   class ClassRegisterAgent
//-----------------------------------------------------------------------


template <class T>

  class ClassRegisterAgent

{
 public:

  inline                    ClassRegisterAgent  ();


 private:

  static
    ClassRegisterAgent      instance_;

};


//-----------------------------------------------------------------------
//   macro JEM_DEFINE_CLASS
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_CLASS
#define JEM_DEFINE_CLASS( Type )                        \
                                                        \
jem::Class* Type::getType ()                            \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
jem::Class* Type::makeClass ()                          \
{                                                       \
  return new jem::ClassTemplate<Self> ();               \
}                                                       \
                                                        \
jem::Class* Type::getClass () const                     \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
const char* Type::CLASS_NAME = #Type


//-----------------------------------------------------------------------
//   macro JEM_DEFINE_SERIABLE_CLASS
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_SERIABLE_CLASS
#define JEM_DEFINE_SERIABLE_CLASS( Type )               \
                                                        \
template class jem::ClassRegisterAgent< Type >;         \
                                                        \
jem::Class* Type::getType ()                            \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
jem::Class* Type::makeClass ()                          \
{                                                       \
  jem::ClassTemplate<Self>* klass =                     \
                                                        \
    new jem::ClassTemplate<Self> ();                    \
                                                        \
  klass->addDefaultCtor ();                             \
                                                        \
  return klass;                                         \
}                                                       \
                                                        \
jem::Class* Type::getClass () const                     \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
const char* Type::CLASS_NAME = #Type


//-----------------------------------------------------------------------
//   macro JEM_DEFINE_CUSTOM_CLASS
//-----------------------------------------------------------------------


#undef  JEM_DEFINE_CUSTOM_CLASS
#define JEM_DEFINE_CUSTOM_CLASS( Type )                 \
                                                        \
template class jem::ClassRegisterAgent< Type >;         \
                                                        \
jem::Class* Type::getType ()                            \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
jem::Class* Type::getClass () const                     \
{                                                       \
  return jem::ClassTemplate<Self>::getInstance ();      \
}                                                       \
                                                        \
const char* Type::CLASS_NAME = #Type




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ClassTemplate
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


template <class T>

  Once   ClassTemplate<T>::once_     = JEM_ONCE_INITIALIZER;

template <class T>

  Class* ClassTemplate<T>::instance_ = nullptr;


//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


template <class T>

  inline ClassTemplate<T>::ClassTemplate () :

    Class ( getSuper_(), T::CLASS_NAME )

{}


template <class T>

  inline ClassTemplate<T>::ClassTemplate ( const String& name ) :

    Class ( name )

{}


template <class T>

  inline ClassTemplate<T>::ClassTemplate

  ( Class*         super,
    const String&  name ) :

    Class ( super, name )

{}


template <class T>

  ClassTemplate<T>::~ClassTemplate ()

{}


//-----------------------------------------------------------------------
//   equalType
//-----------------------------------------------------------------------


template <class T>

  bool ClassTemplate<T>::equalType ( const Ref<Object>& obj ) const

{
  if ( ! obj )
  {
    return false;
  }
  else
  {
    // Use a temporary variable to avoid a warning that "an expression
    // with side effects will be evaluated despite being used as an
    // operand to typeid()".

    Object*  tmp = obj.get ();

    return (typeid(T) == typeid(*tmp));
  }
}


//-----------------------------------------------------------------------
//   addDefaultCtor
//-----------------------------------------------------------------------


template <class T>

  inline void ClassTemplate<T>::addDefaultCtor ()

{
  addConstructor ( ::jem::newInstance< DefaultConstructor<T> > () );
}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


template <class T>

  inline Class* ClassTemplate<T>::getInstance ()

{
  runOnce ( once_, init_ );

  return instance_;
}


//-----------------------------------------------------------------------
//   dynamicCast
//-----------------------------------------------------------------------


template <class T>

  inline T* ClassTemplate<T>::dynamicCast ( Object& rhs )

{
  Class*  self = getInstance  ();
  Class*  type = rhs.getClass ();

  if ( type == self || self->isBaseOf( type ) )
  {
    return static_cast<T*> ( &rhs );
  }

  return (T*) nullptr;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


template <class T>

  void ClassTemplate<T>::init_ ()

{
  instance_ = T::makeClass ();
}


//-----------------------------------------------------------------------
//   template <class T>
//-----------------------------------------------------------------------


template <class T>

  inline Class* ClassTemplate<T>::getSuper_ ()

{
  typedef typename T::Super  SuperClass;

  return SuperClass::getType ();
}



//=======================================================================
//   class ClassRegisterAgent
//=======================================================================


template <class T>

  inline ClassRegisterAgent<T>::ClassRegisterAgent ()

{
  Class::registerClass ( T::getType() );
}


template <class T>

  ClassRegisterAgent<T> ClassRegisterAgent<T>::instance_;


JEM_END_PACKAGE_BASE

#endif
