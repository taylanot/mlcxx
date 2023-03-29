
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

#ifndef JEM_UTIL_TIMER_H
#define JEM_UTIL_TIMER_H

#include <jem/base/Time.h>


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
//   class Timer
//-----------------------------------------------------------------------


class Timer
{
 public:

  typedef Timer         Self;


                        Timer       ()       noexcept;

  void                  swap

    ( Timer&              rhs )              noexcept;

  Time                  time        () const;
  double                toDouble    () const;
  void                  start       ();
  void                  stop        ();
  void                  reset       ();
  bool                  isRunning   () const noexcept;
  static Time           resolution  ();

  void                  readFrom

    ( io::DataInput&      in  );

  void                  writeTo

    ( io::DataOutput&     out )        const;


 private:

  bool                  running_;
  Time                  t0_;
  Time                  dt_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void           decode

  ( Input&                in,
    Timer&                t );

template <class Output>

  inline void           encode

  ( Output&               out,
    const Timer&          t );

template <class Output>

  inline void           print

  ( Output&               out,
    const Timer&          t );

inline void             swap

  ( Timer&                lhs,
    Timer&                rhs ) noexcept;





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void           decode

  ( Input&                in,
    Timer&                t )

{
  t.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void           encode

  ( Output&               out,
    const Timer&          t )

{
  t.writeTo ( out );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void           print

  ( Output&               out,
    const Timer&          t )

{
  print ( out, t.toDouble(), " seconds" );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void             swap

  ( Timer&                lhs,
    Timer&                rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::Timer>
//=======================================================================


template <>

  class TypeTraits<util::Timer> : public BasicTypeTraits

{
 public:

  static const bool IS_SERIALIZABLE = true;


  static String     whatis   ()
  {
    return String ( "a timer" );
  }

  static String     typeName ()
  {
    return String ( "jem::util::Timer" );
  }

};


JEM_END_PACKAGE_BASE

#endif
