
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

#ifndef JEM_UTIL_CPUTIMER_H
#define JEM_UTIL_CPUTIMER_H

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
//   class CPUTimer
//-----------------------------------------------------------------------


class CPUTimer
{
 public:

  typedef CPUTimer      Self;


                        CPUTimer    ();

  void                  swap

    ( CPUTimer&           rhs )              noexcept;

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

    ( io::DataOutput&    out )         const;


 private:

  static void           initOnce_   ();


 private:

  bool                  running_;
  double                t0_;
  double                dt_;

};


//-----------------------------------------------------------------------
//   Related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void           decode

  ( Input&                in,
    CPUTimer&             t );

template <class Output>

  inline void           encode

  ( Output&               out,
    const CPUTimer&       t );

template <class Output>

  inline void           print

  ( Output&               out,
    const CPUTimer&       t );

inline void             swap

  ( CPUTimer&             lhs,
    CPUTimer&             rhs ) noexcept;




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
    CPUTimer&             t )

{
  t.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void           encode

  ( Output&               out,
    const CPUTimer&       t )

{
  t.writeTo ( out );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void           print

  ( Output&               out,
    const CPUTimer&       t )

{
  print ( out, t.toDouble(), " seconds" );
}


//-----------------------------------------------------------------------
//   swap
//-----------------------------------------------------------------------


inline void             swap

  ( CPUTimer&             lhs,
    CPUTimer&             rhs ) noexcept

{
  lhs.swap ( rhs );
}


JEM_END_PACKAGE( util )

JEM_BEGIN_PACKAGE_BASE


//=======================================================================
//   class TypeTraits<util::CPUTimer>
//=======================================================================


template <>

  class TypeTraits<util::CPUTimer> : public BasicTypeTraits

{
 public:

  static const bool IS_SERIALIZABLE = true;


  static String     whatis   ()
  {
    return String ( "a CPU timer" );
  }

  static String     typeName ()
  {
    return String ( "jem::util::CPUTimer" );
  }

};


JEM_END_PACKAGE_BASE

#endif
