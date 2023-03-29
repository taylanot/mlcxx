
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

#ifndef JIVE_UTIL_PRIVATE_NAMELIST_H
#define JIVE_UTIL_PRIVATE_NAMELIST_H

#include <jem/base/String.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/Array.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class NameList
//-----------------------------------------------------------------------


class NameList
{
 public:

  typedef NameList          Self;


                            NameList    ();

  explicit                  NameList

    ( const StringVector&     names );

  inline idx_t              size        () const;
  inline void               clear       ();

  idx_t                     addName

    ( const String&           name );

  inline idx_t              findName

    ( const String&           name )       const;

  inline const String&      getName

    ( idx_t                   index )      const;

  inline StringVector       toArray     ()  const;

  template <class Input>

    inline void             readFrom

    ( Input&                  in );

  template <class Output>

    inline void             writeTo

    ( Output&                 out )        const;


 private:

  idx_t                     linearSearch_

    ( const String&           name )       const;

  idx_t                     binarySearch_

    ( const String&           name )       const;

  void                      sortNames_  ();


 private:

  typedef jem::util
    ::ArrayBuffer<String>   StringBuffer_;


  StringBuffer_             names_;
  IdxVector                 iperm_;
  bool                      isSorted_;

};


// ----------------------------------------------------------------------
//   related functions
// ----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    NameList&                 nl );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const NameList&           nl );





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------


inline idx_t NameList::size () const
{
  return names_.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void NameList::clear ()
{
  names_.clear ();

  isSorted_ = false;
}


//-----------------------------------------------------------------------
//   findName
//-----------------------------------------------------------------------


inline idx_t NameList::findName ( const String& name ) const
{
  if ( names_.size() < 5 )
  {
    return linearSearch_ ( name );
  }
  else
  {
    return binarySearch_ ( name );
  }
}


//-----------------------------------------------------------------------
//   getName
//-----------------------------------------------------------------------


inline const String& NameList::getName ( idx_t index ) const
{
  return names_[index];
}


//-----------------------------------------------------------------------
//   toArray
//-----------------------------------------------------------------------


inline StringVector NameList::toArray () const
{
  return names_.toArray ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


template <class Input>

  inline void NameList::readFrom ( Input& in )

{
  decode ( in, names_ );

  isSorted_ = false;
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


template <class Output>

  inline void NameList::writeTo ( Output& out ) const

{
  encode ( out, names_ );
}


// ----------------------------------------------------------------------
//   decode
// ----------------------------------------------------------------------


template <class Input>

  inline void           decode

  ( Input&                in,
    NameList&             nl )

{
  nl.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void           encode

  ( Output&               out,
    const NameList&       nl )

{
  nl.writeTo ( out );
}


JIVE_END_PACKAGE( util )

#endif
