
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

#ifndef JIVE_UTIL_DATABASESKIN_H
#define JIVE_UTIL_DATABASESKIN_H

#include <utility>
#include <jive/util/import.h>
#include <jive/util/Database.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class DatabaseSkin
//-----------------------------------------------------------------------


class DatabaseSkin
{
 public:

  typedef DatabaseSkin      Self;
  typedef Database          Data;

  typedef IntDBColumn       IntColumn;
  typedef FloatDBColumn     FloatColumn;


  inline Event<Data&>&      newValuesEvent  () const;
  inline Event<Data&>&      newStructEvent  () const;

  inline void               resetEvents     () const;

  template <class Input>

    inline void             readFrom

    ( Input&                  in );

  template <class Output>

    inline void             writeTo

    ( Output&                 out )            const;

  inline bool               operator ==

    ( const Self&             rhs )            const noexcept;

  inline bool               operator ==

    ( const Nil&              rhs )            const noexcept;

  inline bool               operator ==

    ( std::nullptr_t          rhs )            const noexcept;

  inline bool               operator !=

    ( const Self&             rhs )            const noexcept;

  inline bool               operator !=

    ( const Nil&              rhs )            const noexcept;

  inline bool               operator !=

    ( std::nullptr_t          rhs )            const noexcept;

  explicit inline           operator bool   () const noexcept;
  inline bool               operator  !     () const noexcept;

  inline Data*              getData         () const noexcept;
  inline void               clearData       ();
  inline void               clearAll        ();
  inline void               trimToSize      ();

  inline void               store

    ( const Properties&       globdat )        const;

  inline void               store

    ( const String&           name,
      const Properties&       globdat )        const;

  static void               dataError

    ( const String&           context,
      const String&           itemKind,
      const Ref<Data>&        data );


 protected:

  inline                    DatabaseSkin    ()       noexcept;

  explicit inline           DatabaseSkin

    ( const Ref<Data>&        data )                 noexcept;

  explicit inline           DatabaseSkin

    ( Ref<Data>&&             data )                 noexcept;

  inline                    DatabaseSkin

    ( const Self&             rhs )                  noexcept;

  inline                    DatabaseSkin

    ( Self&&                  rhs )                  noexcept;

  inline void               assign_

    ( const Self&             rhs );

  inline void               swap_

    ( Self&                   rhs )                  noexcept;


 protected:

  Ref<Data>                 data_;


 private:

  Self&                     operator = ( const Self& );

};


//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


template <class Input>

  inline void               decode

  ( Input&                    in,
    DatabaseSkin&             dbase );

template <class Output>

  inline void               encode

  ( Output&                   out,
    const DatabaseSkin&       dbase );

template <class Output>

  inline void               print

  ( Output&                   out,
    const DatabaseSkin&       dbase );

inline bool                 operator ==

  ( const Nil&                lhs,
    const DatabaseSkin&       rhs )                  noexcept;

inline bool                 operator ==

  ( std::nullptr_t            lhs,
    const DatabaseSkin&       rhs )                  noexcept;

inline bool                 operator !=

  ( const Nil&                lhs,
    const DatabaseSkin&       rhs )                  noexcept;

inline bool                 operator !=

  ( std::nullptr_t            lhs,
    const DatabaseSkin&       rhs )                  noexcept;





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class DatabaseSkin
//=======================================================================

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline DatabaseSkin::DatabaseSkin () noexcept
{}


inline DatabaseSkin::DatabaseSkin

  ( const Ref<Data>&  data ) noexcept :

    data_ ( data )

{}


inline DatabaseSkin::DatabaseSkin

  ( Ref<Data>&&  data ) noexcept :

    data_ ( std::move( data ) )

{}


inline DatabaseSkin::DatabaseSkin ( const Self& rhs ) noexcept :

  data_ ( rhs.data_ )

{}


inline DatabaseSkin::DatabaseSkin ( Self&& rhs ) noexcept :

  data_ ( std::move( rhs.data_ ) )

{}


//-----------------------------------------------------------------------
//   newValuesEvent
//-----------------------------------------------------------------------


inline Event<Database&>& DatabaseSkin::newValuesEvent () const
{
  return data_->newValuesEvent;
}


//-----------------------------------------------------------------------
//   newStructEvent
//-----------------------------------------------------------------------


inline Event<Database&>& DatabaseSkin::newStructEvent () const
{
  return data_->newStructEvent;
}


//-----------------------------------------------------------------------
//   resetEvents
//-----------------------------------------------------------------------


inline void DatabaseSkin::resetEvents () const
{
  data_->resetEvents ();
}


//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------


template <class Input>

  inline void DatabaseSkin::readFrom ( Input& in )

{
  decode ( in, data_ );
}


//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------


template <class Output>

  inline void DatabaseSkin::writeTo ( Output& out ) const

{
  encode ( out, data_ );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool
  DatabaseSkin::operator == ( const Self& rhs ) const noexcept
{
  return (data_ == rhs.data_);
}


inline bool
  DatabaseSkin::operator == ( const Nil& rhs ) const noexcept
{
  return (data_ == rhs);
}


inline bool
  DatabaseSkin::operator == ( std::nullptr_t rhs ) const noexcept
{
  return (data_ == rhs);
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool
  DatabaseSkin::operator != ( const Self& rhs ) const noexcept
{
  return (data_ != rhs.data_);
}


inline bool
  DatabaseSkin::operator != ( const Nil& rhs ) const noexcept
{
  return (data_ != rhs);
}


inline bool
  DatabaseSkin::operator != ( std::nullptr_t rhs ) const noexcept
{
  return (data_ != rhs);
}


//-----------------------------------------------------------------------
//   operator bool
//-----------------------------------------------------------------------


inline DatabaseSkin::operator bool () const noexcept
{
  return (data_ != nullptr);
}


inline bool DatabaseSkin::operator ! () const noexcept
{
  return (data_ == nullptr);
}



//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------


inline Database* DatabaseSkin::getData () const noexcept
{
  return data_.get ();
}


//-----------------------------------------------------------------------
//   clearData
//-----------------------------------------------------------------------


inline void DatabaseSkin::clearData ()
{
  data_->clearData ();
}


//-----------------------------------------------------------------------
//   clearAll
//-----------------------------------------------------------------------


inline void DatabaseSkin::clearAll ()
{
  data_->clearAll ();
}


//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------


inline void DatabaseSkin::trimToSize ()
{
  data_->trimToSize ();
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void DatabaseSkin::store

  ( const Properties&  globdat ) const

{
  data_->store ( globdat );
}


inline void DatabaseSkin::store

  ( const String&      name,
    const Properties&  globdat ) const

{
  data_->store ( name, globdat );
}


//-----------------------------------------------------------------------
//   assign_
//-----------------------------------------------------------------------


inline void DatabaseSkin::assign_ ( const Self& rhs )
{
  data_ = rhs.data_;
}


//-----------------------------------------------------------------------
//   swap_
//-----------------------------------------------------------------------


inline void DatabaseSkin::swap_ ( Self& rhs ) noexcept
{
  data_.swap ( rhs.data_ );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   decode
//-----------------------------------------------------------------------


template <class Input>

  inline void             decode

  ( Input&                  in,
    DatabaseSkin&           dbase )

{
  dbase.readFrom ( in );
}


//-----------------------------------------------------------------------
//   encode
//-----------------------------------------------------------------------


template <class Output>

  inline void             encode

  ( Output&                 out,
    const DatabaseSkin&     dbase )

{
  dbase.writeTo ( out );
}


//-----------------------------------------------------------------------
//   print
//-----------------------------------------------------------------------


template <class Output>

  inline void             print

  ( Output&                 out,
    const DatabaseSkin&     dbase )

{
  dbase.getData()->printTo ( out );
}


//-----------------------------------------------------------------------
//   operator ==
//-----------------------------------------------------------------------


inline bool               operator ==

  ( const Nil&              lhs,
    const DatabaseSkin&     rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


inline bool               operator ==

  ( std::nullptr_t          lhs,
    const DatabaseSkin&     rhs ) noexcept

{
  return rhs.operator == ( lhs );
}


//-----------------------------------------------------------------------
//   operator !=
//-----------------------------------------------------------------------


inline bool               operator !=

  ( const Nil&              lhs,
    const DatabaseSkin&     rhs ) noexcept

{
  return rhs.operator != ( lhs );
}


inline bool               operator !=

  ( std::nullptr_t          lhs,
    const DatabaseSkin&     rhs ) noexcept

{
  return rhs.operator != ( lhs );
}


JIVE_END_PACKAGE( util )

#endif
