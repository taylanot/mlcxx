
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

#ifndef JEM_STD_COMPLEX_H
#define JEM_STD_COMPLEX_H

#include <complex>
#include <jem/base/TypeTraits.h>


#undef  JEM_STD_COMPLEX
#define JEM_STD_COMPLEX


JEM_BEGIN_PACKAGE( io )


//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input, class T>

  inline void             decode

  ( Input&                  in,
    std::complex<T>&        z )

{
  T  re, im;

  decode ( in, re, im );

  z = std::complex<T> ( re, im );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void             encode

  ( Output&                 out,
    const std::complex<T>&  z )

{
  encode ( out, z.real(), z.imag() );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output, class T>

  inline void             print

  ( Output&                 out,
    const std::complex<T>&  z )

{
  print ( out, '(', z.real(), ',', z.imag(), ')' );
}


JEM_END_PACKAGE( io )

JEM_BEGIN_PACKAGE_BASE


//-----------------------------------------------------------------------
//   TypeTraits < std::complex >
//-----------------------------------------------------------------------


template <class T>
  class TypeTraits< std::complex<T> > : public BasicTypeTraits
{
 public:

  static const bool         IS_SERIALIZABLE  =

    IsSerializable<T>::VALUE;


  static long               hashValue

    ( const std::complex<T>&  z )

  {
    return ( TypeTraits<T>::hashValue( z.real() ) ^
             TypeTraits<T>::hashValue( z.imag() ) );
  }

  static String             typeName ()
  {
    return ( String ( "std::complex<" ) +
             TypeTraits<T>::typeName () +
             String ( '>' ) );
  }

};


JEM_END_PACKAGE_BASE

#endif
