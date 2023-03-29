
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

#ifndef JEM_IO_WRITER_H
#define JEM_IO_WRITER_H

#include <jem/base/Object.h>
#include <jem/io/TextOutput.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class Writer
//-----------------------------------------------------------------------


class Writer : public Object,
               public TextOutput
{
 public:

  JEM_DECLARE_CLASS   ( Writer, Object );


  virtual Ref<Writer>   dup     ();
  virtual void          close   ();
  virtual void          flush   ();

  virtual void          write

    ( int                 c )      override = 0;

  virtual void          write

    ( int                 c,
      idx_t               n )      override;

  virtual void          write

    ( const char*         buf,
      idx_t               n )      override;

  virtual void          writeNoThrow

    ( const char*         buf,
      idx_t               n )      noexcept;

  virtual void          printByte

    ( byte                b )      override;

  virtual void          printBool

    ( bool                b )      override;

  virtual void          printInt

    ( lint                i )      override;

  virtual void          printFloat

    ( double              d )      override;

  template <class T>
    inline Self&        operator <<

    ( const T&            obj );


 protected:

  virtual              ~Writer  ();


 private:

  enum                { BUFSIZE_ = 128 };

};


//-----------------------------------------------------------------------
//   related classes
//-----------------------------------------------------------------------


struct                  Flush   {};
extern const Flush      flush;


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Output>

  inline void           print

  ( Output&               out,
    const Flush&          fl );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   operator <<
//-----------------------------------------------------------------------


template <class T>

  inline Writer&        Writer::operator <<

  ( const T&              obj )

{
  print ( *this, obj );
  return *this;
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void           print

  ( Output&               out,
    const Flush&          fl )

{
  out.flush ();
}


JEM_END_PACKAGE( io )

#endif
