
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
#include <jem/base/Thread.h>
#include <jem/io/TermInput.h>
#include <jem/io/StdTermReader.h>


JEM_BEGIN_PACKAGE( io )


//=======================================================================
//   class TermInput::Data_
//=======================================================================


class TermInput::Data_
{
 public:

  static inline int       getKey     ();
  static TermReader*      getReader  ();


 private:

  static void             init_      ();


 private:

  static Once             once_;
  static int              key_;

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


Once  TermInput::Data_::once_ = JEM_ONCE_INITIALIZER;
int   TermInput::Data_::key_  = Thread::NULL_KEY;


//-----------------------------------------------------------------------
//   getKey
//-----------------------------------------------------------------------


inline int TermInput::Data_::getKey ()
{
  runOnce ( once_, init_ );

  return key_;
}


//-----------------------------------------------------------------------
//   getReader
//-----------------------------------------------------------------------


TermReader* TermInput::Data_::getReader ()
{
  const int    key    = getKey ();

  TermReader*  reader =

    staticCast<TermReader*> ( Thread::getLocal( key ) );

  if ( ! reader )
  {
    Ref<TermReader>  tmp = newInstance<StdTermReader> ();

    Thread::setLocal ( key, tmp );

    reader = tmp.get ();
  }

  return reader;
}


//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void TermInput::Data_::init_ ()
{
  key_ = Thread::newLocalKey ();
}


//=======================================================================
//   class TermInput
//=======================================================================

//-----------------------------------------------------------------------
//   poll
//-----------------------------------------------------------------------


idx_t TermInput::poll ( const Time& timeout )
{
  return Data_::getReader()->poll ( timeout );
}


//-----------------------------------------------------------------------
//   readLine
//-----------------------------------------------------------------------


String TermInput::readLine ( const String& prompt )
{
  return Data_::getReader()->readLine ( prompt, 0 );
}


String TermInput::readLine

  ( const String&   prompt,
    WordCompleter&  wcomp )

{
  return Data_::getReader()->readLine ( prompt, &wcomp );
}


//-----------------------------------------------------------------------
//   clearHistory
//-----------------------------------------------------------------------


void TermInput::clearHistory ()
{
  return Data_::getReader()->clearHistory ();
}


//-----------------------------------------------------------------------
//   getReader
//-----------------------------------------------------------------------


TermReader* TermInput::getReader ()
{
  return Data_::getReader ();
}


//-----------------------------------------------------------------------
//   setReader
//-----------------------------------------------------------------------


void TermInput::setReader ( Ref<TermReader> reader )
{
  JEM_PRECHECK ( reader );

  const int  key = Data_::getKey ();

  Thread::setLocal ( key, reader );
}


JEM_END_PACKAGE( io )
