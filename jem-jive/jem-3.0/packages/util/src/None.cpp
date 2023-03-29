
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


#include <jem/base/Once.h>
#include <jem/base/Class.h>
#include <jem/base/Clonable.h>
#include <jem/base/reflect/Constructor.h>
#include <jem/io/Serializable.h>
#include <jem/util/None.h>


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class NoneCtor
//=======================================================================


class NoneCtor : public Constructor
{
 public:

  inline                  NoneCtor        ();

  virtual idx_t           argCount        () const override;

  virtual void            getArgTypes

    ( Class**               types )          const override;

  virtual Ref<Object>     call

    ( const Ref<Object>*    args )           const override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NoneCtor::NoneCtor ()
{}


//-----------------------------------------------------------------------
//   argCount
//-----------------------------------------------------------------------


idx_t NoneCtor::argCount () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   getArgTypes
//-----------------------------------------------------------------------


void NoneCtor::getArgTypes ( Class** types ) const
{}


//-----------------------------------------------------------------------
//   call
//-----------------------------------------------------------------------


Ref<Object> NoneCtor::call ( const Ref<Object>* args ) const
{
  return None::getInstance ();
}


//=======================================================================
//   class NoneClass
//=======================================================================


class NoneClass : public Class
{
 public:

  typedef Class           Super;


  inline                  NoneClass   ();

  virtual bool            isInstance

    ( const Ref<Object>&    obj )        const;

  virtual bool            equalType

    ( const Ref<Object>&    obj )        const override;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NoneClass::NoneClass () :

  Super ( Object::getType(), "jem::util::None" )

{
  setWhatis      ( "none" );
  addConstructor ( jem::newInstance<NoneCtor>() );
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool NoneClass::isInstance ( const Ref<Object>& obj ) const
{
  return isNone ( obj );
}


//-----------------------------------------------------------------------
//   equalType
//-----------------------------------------------------------------------


bool NoneClass::equalType ( const Ref<Object>& obj ) const
{
  return isNone ( obj );
}


//=======================================================================
//   class NoneObject
//=======================================================================


class NoneObject : public Object,
                   public Clonable,
                   public io::Serializable
{
 public:

  typedef NoneObject        Self;
  typedef Object            Super;


  inline                    NoneObject    ();

  static inline Class*      getType       ();
  virtual Class*            getClass      () const override;
  virtual String            toString      () const override;
  virtual lint              hashValue     () const override;

  virtual bool              equals

    ( const Ref<Object>&      rhs )          const override;

  virtual Ref<Object>       clone         () const override;

  virtual void              readFrom

    ( io::ObjectInput&        in )                 override;

  virtual void              writeTo

    ( io::ObjectOutput&       out )          const override;

  static inline Self*       getInstance   ();


 private:

  static void               init_         ();


 private:


  NoneClass                 myClass_;

  static Once               once_;
  static NoneObject*        instance_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Once         NoneObject::once_     = JEM_ONCE_INITIALIZER;
NoneObject*  NoneObject::instance_ = 0;


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline NoneObject::NoneObject ()
{}


//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


inline Class* NoneObject::getType ()
{
  return & getInstance()->myClass_;
}


//-----------------------------------------------------------------------
//   getClass
//-----------------------------------------------------------------------


Class* NoneObject::getClass () const
{
  return const_cast<NoneClass*>( & myClass_ );
}


//-----------------------------------------------------------------------
//   toString
//-----------------------------------------------------------------------


String NoneObject::toString () const
{
  return "none";
}


//-----------------------------------------------------------------------
//   hashValue
//-----------------------------------------------------------------------


lint NoneObject::hashValue () const
{
  return 0;
}


//-----------------------------------------------------------------------
//   equals
//-----------------------------------------------------------------------


bool NoneObject::equals ( const Ref<Object>& rhs ) const
{
  return (rhs == this);
}


//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------


Ref<Object> NoneObject::clone () const
{
  Ref<Object>  self ( const_cast<Self*>( this ) );

  return self;
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


void NoneObject::readFrom ( io::ObjectInput& in )
{}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


void NoneObject::writeTo ( io::ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


inline NoneObject* NoneObject::getInstance ()
{
  runOnce ( once_, init_ );

  return instance_;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void NoneObject::init_ ()
{
  Ref<NoneObject>  tmp =  newInstance<NoneObject> ();

  instance_ = tmp.get ();

  Collectable::addRef ( *tmp );
}


//=======================================================================
//   class NoneInit
//=======================================================================


class NoneInit
{
 public:

  NoneInit ()
  {
    Class::registerClass ( NoneObject::getType() );
  }

};


static NoneInit  noneInit;


//=======================================================================
//   class None
//=======================================================================

//-----------------------------------------------------------------------
//   getType
//-----------------------------------------------------------------------


Class* None::getType ()
{
  return NoneObject::getType ();
}


//-----------------------------------------------------------------------
//   getInstance
//-----------------------------------------------------------------------


Ref<Object> None::getInstance ()
{
  return NoneObject::getInstance ();
}


//-----------------------------------------------------------------------
//   isInstance
//-----------------------------------------------------------------------


bool None::isInstance ( Object& obj )
{
  return (NoneObject::getInstance() == &obj);
}


JEM_END_PACKAGE( util )
