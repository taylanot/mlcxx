
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

#ifndef JEM_IO_NUMBERFORMATTER_H
#define JEM_IO_NUMBERFORMATTER_H

#include <jem/io/NumberFormat.h>


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   class NumberFormatter
//-----------------------------------------------------------------------


class NumberFormatter
{
 public:

  typedef NumberFormatter   Self;


                            NumberFormatter ();

  explicit                  NumberFormatter

    ( const NumberFormat&     fmt );

                           ~NumberFormatter ();

  inline String             operator        ()

    ( byte                    val );

  inline String             operator        ()

    ( short                   val );

  inline String             operator        ()

    ( int                     val );

  inline String             operator        ()

    ( long                    val );

  inline String             operator        ()

    ( lint                    val );

  inline String             operator        ()

    ( idx_t                   val );

  inline String             operator        ()

    ( float                   val );

  inline String             operator        ()

    ( double                  val );


 public:

  NumberFormat              nformat;


 private:

                            NumberFormatter ( const Self& );
  Self&                     operator =      ( const Self& );

  void                      init_           ();


 private:

  char*                     buffer_;

};






//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   operator ()
//-----------------------------------------------------------------------


inline String NumberFormatter::operator () ( byte val )
{
  const int  n = nformat.print ( buffer_, (lint) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( short val )
{
  const int  n = nformat.print ( buffer_, (lint) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( int val )
{
  const int  n = nformat.print ( buffer_, (lint) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( long val )
{
  const int  n = nformat.print ( buffer_, (lint) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( lint val )
{
  const int  n = nformat.print ( buffer_, val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( idx_t val )
{
  const int  n = nformat.print ( buffer_, (lint) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( float val )
{
  const int  n = nformat.print ( buffer_, (double) val );

  return String ( buffer_, buffer_ + n );
}


inline String NumberFormatter::operator () ( double val )
{
  const int  n = nformat.print ( buffer_, val );

  return String ( buffer_, buffer_ + n );
}


JEM_END_PACKAGE( io )

#endif
