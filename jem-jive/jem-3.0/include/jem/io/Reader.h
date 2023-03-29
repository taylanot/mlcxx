
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

#ifndef JEM_IO_READER_H
#define JEM_IO_READER_H

#include <jem/base/Object.h>
#include <jem/io/TextInput.h>


namespace jem
{
  class Time;
}


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Reader
//-----------------------------------------------------------------------


class Reader : public Object,
               public TextInput
{
 public:

  JEM_DECLARE_CLASS   ( Reader, Object );


  virtual Ref<Reader>   dup           ();
  virtual void          close         ();

  virtual idx_t         poll

    ( const Time&         timeout );

  virtual int           read          () override = 0;

  virtual idx_t         read

    ( char*               buf,
      idx_t               n )            override;

  virtual idx_t         readUntil

    ( char*               buf,
      idx_t               n,
      int                 delim );

  virtual void          pushBack

    ( int                 c )            override = 0;

  virtual void          skipUntil

    ( int                 delim );

  virtual void          skipWhite     () override;

  virtual idx_t         skip

    ( idx_t               n );

  String                readLine      ();
  virtual byte          parseByte     () override;
  virtual bool          parseBool     () override;
  virtual char          parseChar     () override;
  virtual short         parseShort    () override;
  virtual int           parseInt      () override;
  virtual long          parseLong     () override;
  virtual lint          parseLint     () override;
  virtual float         parseFloat    () override;
  virtual double        parseDouble   () override;

  template <class T>
    inline Self&        operator >>

    ( T&                  obj );


 protected:

  virtual              ~Reader        ();


 private:

  static const int      BUFSIZE_;

  bool                  matchIgnoreCase_

    ( const char*         str );

};


//#######################################################################
//  implementation
//#######################################################################

//-----------------------------------------------------------------------
//   operator >>
//-----------------------------------------------------------------------


template <class T>

  inline void           readFromReader

  ( Reader&               in,
    T&                    obj )

{
  read ( in, obj );
}


template <class T>

  inline Reader&        Reader::operator >>

  ( T&                    obj )

{
  // The non-member function read() cannot be called directly because it
  // is hidden by the member function read().

  jem::io::readFromReader ( *this, obj );

  return *this;
}


JEM_END_PACKAGE( io )

#endif
