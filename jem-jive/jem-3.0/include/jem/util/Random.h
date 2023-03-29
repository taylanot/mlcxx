
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

#ifndef JEM_UTIL_RANDOM_H
#define JEM_UTIL_RANDOM_H

#include <jem/base/TypeTraits.h>


namespace jem
{
  namespace io
  {
    class DataInput;
    class DataOutput;
  }
}


JEM_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class Random
//-----------------------------------------------------------------------


class Random
{
 public:

  static const int        MAX_VALUE;


  explicit                Random

    ( int                   seed = 0 );

                          Random

    ( const Random&         rhs );

                          Random

    ( Random&&              rhs );

                         ~Random        ();

  Random&                 operator =

    ( const Random&         rhs );

  Random&                 operator =

    ( Random&&              rhs )                noexcept;

  void                    swap

    ( Random&               rhs )                noexcept;

  void                    restart

    ( int                   seed = 0 );

  double                  next          ();

  idx_t                   next

    ( idx_t                 ubound );

  int                     nextValue     ();
  double                  nextGaussian  ();

  void                    readFrom

    ( io::DataInput&        in  );

  void                    writeTo

    ( io::DataOutput&       out )          const;

 private:

  void                    copy_

    ( const Random&         rhs );


 private:

  static const int        COEFFC_;
  static const int        LARGE_;
  static const int        SEED_;

  int*                    coeffs_;
  int                     ipos_;
  int                     jpos_;
  int                     igauss_;
  double                  xgauss_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    Random&                 rand );

template <class Output>

  inline void             encode

  ( Output&                 out,
    const Random&           rand );



//#######################################################################
//  Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    Random&                 rand )

{
  rand.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void             encode

  ( Output&                 out,
    const Random&           rand )

{
  rand.writeTo ( out );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::Random>
//=======================================================================


template <>

  class TypeTraits<util::Random> : public BasicTypeTraits

{
 public:

  static const bool IS_SERIALIZABLE = true;


  static String     whatis   ()
  {
    return String ( "a random number generator" );
  }

  static String     typeName ()
  {
    return String ( "jem::util::Random" );
  }

};


JEM_END_PACKAGE_BASE

#endif
