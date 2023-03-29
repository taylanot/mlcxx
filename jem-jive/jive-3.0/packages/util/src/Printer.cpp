
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


#include <jem/base/assert.h>
#include <jem/base/Once.h>
#include <jem/base/Thread.h>
#include <jem/base/System.h>
#include <jem/io/PrintWriter.h>
#include <jive/util/Printer.h>


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::newInstance;
using jem::Thread;
using jem::System;


//=======================================================================
//   class Printer::Flusher_
//=======================================================================


class Printer::Flusher_
{
 public:

                   ~Flusher_ ();

  static Flusher_   instance;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Printer::Flusher_ Printer::Flusher_::instance;


//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


Printer::Flusher_::~Flusher_ ()
{
  PrintWriter*  p =

    staticCast<PrintWriter*> ( Thread::getLocal ( getKey_() ) );

  if ( p )
  {
    try
    {
      p->flush ();
    }
    catch ( ... ) {}
  }
}


//=======================================================================
//   class Printer
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


int  Printer::key_ = Thread::NULL_KEY;


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


PrintWriter& Printer::get ()
{
  const int     key = getKey_ ();

  PrintWriter*  p   =

    staticCast<PrintWriter*> ( Thread::getLocal( key ) );

  if ( ! p )
  {
    Ref<PrintWriter>  tmp =

      newInstance<PrintWriter> ( & System::out() );

    tmp->setOptions  ( PrintWriter::FLUSH_LINES );
    Thread::setLocal ( key, tmp );

    p = tmp.get ();
  }

  return * p;
}


//-----------------------------------------------------------------------
//   set
//-----------------------------------------------------------------------


void Printer::set ( Ref<PrintWriter> out )
{
  JEM_PRECHECK2 ( out, "NULL PrintWriter" );

  Thread::setLocal ( getKey_(), out );
}


//-----------------------------------------------------------------------
//   flush
//-----------------------------------------------------------------------


void Printer::flush ()
{
  get().flush ();
}


//-----------------------------------------------------------------------
//   getKey_
//-----------------------------------------------------------------------


int Printer::getKey_ ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, & newKey_ );

  return key_;
}


//-----------------------------------------------------------------------
//   newKey_
//-----------------------------------------------------------------------


void Printer::newKey_ ()
{
  key_ = Thread::newLocalKey ();
}


JIVE_END_PACKAGE( util )
