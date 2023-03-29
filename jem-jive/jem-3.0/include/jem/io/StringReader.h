
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

#ifndef JEM_IO_STRINGREADER_H
#define JEM_IO_STRINGREADER_H

#include <jem/base/String.h>
#include <jem/io/Reader.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class StringReader
//-----------------------------------------------------------------------


class StringReader : public Reader
{
 public:

  JEM_DECLARE_CLASS     ( StringReader, Reader );


                          StringReader  ();

  explicit                StringReader

    ( const String&         str );

  explicit                StringReader

    ( String&&              str );

                          StringReader

    ( const Self&           rhs );

  virtual Ref<Reader>     dup           () override;

  virtual idx_t           poll

    ( const Time&           timeout )      override;

  virtual int             read          () override;

  virtual idx_t           read

    ( char*                 buf,
      idx_t                 n )            override;

  virtual idx_t           readUntil

    ( char*                 buf,
      idx_t                 n,
      int                   delim )        override;

  virtual void            pushBack

    ( int                   c )            override;

  virtual void            skipUntil

    ( int                   delim )        override;

  virtual void            skipWhite     () override;

  virtual idx_t           skip

    ( idx_t                 n )            override;

  void                    setString

    ( const String&         str );

  void                    reset         ();


 protected:

  virtual                ~StringReader   ();


 private:

  String                  str_;
  idx_t                   pos_;

};


JEM_END_PACKAGE( io )

#endif
