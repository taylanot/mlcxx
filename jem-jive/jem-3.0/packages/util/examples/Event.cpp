
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

/************************************************************************
 *
 *  This example shows how to use the Event class
 *
 ************************************************************************/


#include <jem/base/System.h>
#include <jem/util/Event.h>


using namespace jem;
using namespace jem::io;
using namespace jem::util;


//-----------------------------------------------------------------------
//   printValue
//-----------------------------------------------------------------------


void printValue ( int i )
{
  print ( System::out(), i, '\n' );
}


//-----------------------------------------------------------------------
//   class Listener
//-----------------------------------------------------------------------


class Listener : public Object
{
 public:

                      Listener    ();

  void                printValue

    ( double            f );

  void                printHello  ();


 protected:

  virtual            ~Listener    ();

};


Listener::Listener ()
{}


Listener::~Listener ()
{}


void Listener::printValue ( double f )
{
  print ( System::out(), f, endl );
}


void Listener::printHello ()
{
  print ( System::out(), "Hello\n" );
}


//-----------------------------------------------------------------------
//   class Signaller
//-----------------------------------------------------------------------


class Signaller : public Object
{
 public:

  Event<>             zeroReached;
  Event<int,int>      newValue;

                      Signaller ();

  void                setAndDecr

    ( int               i );

  void                destroy   ();


  static
    Ref<Signaller>    instance;


 protected:

  virtual            ~Signaller ();


 private:

  int                 value_;

};


Ref<Signaller> Signaller::instance;


Signaller::Signaller () : value_ ( 0 )
{}


Signaller::~Signaller ()
{
  print ( System::out(), "Bye Bye ...\n" );
}


void Signaller::setAndDecr ( int i )
{
  int oldValue = value_;

  value_ = (--i);

  if ( value_ <= 0 )
  {
    zeroReached.emit ();
  }
  else
  {
    newValue.emit ( value_, oldValue );
  }
}


void Signaller::destroy ()
{
  instance = NIL;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run()
{
  Ref<Listener>  listener = newInstance<Listener>  ();

  Signaller::instance     = newInstance<Signaller> ();

  {
    Ref<Signaller>  s = Signaller::instance;

    connect ( s->zeroReached, listener, & Listener::printHello  );
    connect ( s->zeroReached, s,        & Signaller::destroy    );
    connect ( s->newValue,    s,        & Signaller::setAndDecr );
    connect ( s->newValue,    listener, & Listener::printValue  );
    connect ( s->newValue,              & printValue            );
  }

  Signaller::instance->setAndDecr ( 10 );

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}
