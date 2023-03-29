
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
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
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

/************************************************************************

  This example program demonstrates how to use the garbage collection
  framework implemented by Jem. It shows how to defines a collectable
  class; how to create instances of a collectable class; and how to
  access collectable objects through Ref objects.

************************************************************************/

#include <jem/base/System.h>
#include <jem/base/Collectable.h>


//-----------------------------------------------------------------------

class Foo : public jem::Collectable
{
 public:

  explicit    Foo ( int n );


  int         value;

 private:
 
  virtual    ~Foo ();

};


Foo::Foo ( int n ) : value ( n )
{
  jem::System::out() << "Foo::Foo  ( " << value << " )\n";
}


Foo::~Foo ()
{
  jem::System::out() << "Foo::~Foo ( " << value << " )\n";
}


//-----------------------------------------------------------------------

int run ()
{
  jem::Ref<Foo>  first;
  jem::Ref<Foo>  second;

  first        = jem::newInstance<Foo> ( 1 );
  second       = jem::newInstance<Foo> ( 2 );

  first        = second;
  first->value = 3;

  return 0;
}


//-----------------------------------------------------------------------

int main ()
{
  return jem::System::exec ( & run );
}
