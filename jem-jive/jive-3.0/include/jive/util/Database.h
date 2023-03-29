
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

#ifndef JIVE_UTIL_DATABASE_H
#define JIVE_UTIL_DATABASE_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/util/import.h>
#include <jive/util/EventSource.h>


JIVE_BEGIN_PACKAGE( util )


using jem::Tuple;

class ItemSet;
class Database;


//-----------------------------------------------------------------------
//   class DBColumn
//-----------------------------------------------------------------------


class DBColumn
{
 public:

  virtual String            getName         () const = 0;
  virtual Database*         getDatabase     () const = 0;
  virtual ItemSet*          getRowItems     () const = 0;
  String                    getContext      () const;

  String                    getContext

    ( idx_t                   irow )           const;

  idx_t                     rowCount        () const;

  virtual idx_t             rowSize

    ( idx_t                   irow )           const = 0;

  virtual idx_t             maxRowSize      () const;
  virtual IdxVector         getNonEmptyRows () const = 0;
  virtual void              clear           ();

  virtual void              clearRow

    ( idx_t                   irow )                 = 0;

  virtual void              trimToSize      ();

  virtual void              printTo

    ( PrintWriter&            out )            const = 0;


 protected:

                            DBColumn        ();
  virtual                  ~DBColumn        ();


 private:

                            DBColumn   ( const DBColumn& );
  DBColumn&                 operator = ( const DBColumn& );

};


//-----------------------------------------------------------------------
//   class IntDBColumn
//-----------------------------------------------------------------------


class IntDBColumn : virtual public DBColumn
{
 public:

  typedef idx_t             ValueType;


  virtual idx_t             getData

    ( idx_t*                  buf,
      idx_t                   size,
      idx_t                   irow )           const = 0;

  virtual void              setData

    ( idx_t                   irow,
      const idx_t*            buf,
      idx_t                   count )                = 0;

  virtual void              printTo

    ( PrintWriter&            out )            const override;

  inline bool               findValue

    ( idx_t&                  value,
      idx_t                   irow )           const;

  idx_t                     getValue

    ( idx_t                   irow )           const;

  idx_t                     findValues

    ( const IdxVector&        buf,
      idx_t                   count,
      idx_t                   irow )           const;

  inline idx_t              findValues

    ( const IdxVector&        buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const IdxVector&        buf,
      idx_t                   count,
      idx_t                   irow )           const;

  inline void               getValues

    ( const IdxVector&        buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const IdxMatrix&        buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const IdxCubix&         buf,
      idx_t                   irow )           const;

  template <int M, int N>

    inline void             getValues

    ( Tuple<idx_t,M,N>&       buf,
      idx_t                   irow )           const;

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   value );

  void                      setValues

    ( idx_t                   irow,
      const IdxVector&        buf,
      idx_t                   count );

  inline void               setValues

    ( idx_t                   irow,
      const IdxVector&        buf );

  void                      setValues

    ( idx_t                   irow,
      const IdxMatrix&        buf );

  void                      setValues

    ( idx_t                   irow,
      const IdxCubix&         buf );

  template <int M, int N>

    inline void             setValues

    ( idx_t                   irow,
      const Tuple
        <idx_t,M,N>&          buf );



 protected:

  virtual                  ~IntDBColumn     ();


 private:

  void                      missingDataError_

    ( idx_t                   irow,
      idx_t                   actualCount,
      idx_t                   formalCount )    const;

};


//-----------------------------------------------------------------------
//   class FloatDBColumn
//-----------------------------------------------------------------------


class FloatDBColumn : virtual public DBColumn
{
 public:

  typedef double            ValueType;


  virtual idx_t             getData

    ( double*                 buf,
      idx_t                   size,
      idx_t                   irow )           const = 0;

  virtual void              setData

    ( idx_t                   irow,
      const double*           buf,
      idx_t                   count )                = 0;

  virtual void              printTo

    ( PrintWriter&            out )            const override;

  inline bool               findValue

    ( double&                 value,
      idx_t                   irow )           const;

  double                    getValue

    ( idx_t                   irow )           const;

  idx_t                     findValues

    ( const Vector&           buf,
      idx_t                   count,
      idx_t                   irow )           const;

  inline idx_t              findValues

    ( const Vector&           buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const Vector&           buf,
      idx_t                   count,
      idx_t                   irow )           const;

  inline void               getValues

    ( const Vector&           buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const Matrix&           buf,
      idx_t                   irow )           const;

  void                      getValues

    ( const Cubix&            buf,
      idx_t                   irow )           const;

  template <int M, int N>

    inline void             getValues

    ( Tuple<double,M,N>&      buf,
      idx_t                   irow )           const;

  inline void               setValue

    ( idx_t                   irow,
      double                  value );

  void                      setValues

    ( idx_t                   irow,
      const Vector&           buf,
      idx_t                   count );

  inline void               setValues

    ( idx_t                   irow,
      const Vector&           buf );

  void                      setValues

    ( idx_t                   irow,
      const Matrix&           buf );

  void                      setValues

    ( idx_t                   irow,
      const Cubix&            buf );

  template <int M, int N>

    inline void             setValues

    ( idx_t                   irow,
      const Tuple
      <double,M,N>&           buf );


 protected:

  virtual                  ~FloatDBColumn   ();


 private:

  void                      missingDataError_

    ( idx_t                   irow,
      idx_t                   actualCount,
      idx_t                   formalCount )    const;

};


//-----------------------------------------------------------------------
//   class Database
//-----------------------------------------------------------------------


class Database : public NamedObject,
                 public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( Database, NamedObject );

  typedef IntDBColumn       IntColumn;
  typedef FloatDBColumn     FloatColumn;

  typedef jem::Array
    <IntColumn*>            IntColArray;
  typedef jem::Array
    <FloatColumn*>          FloatColArray;

  static const char*        MAIN_DB;

  Event < Self& >           newValuesEvent;
  Event < Self& >           newStructEvent;


  virtual String            getContext      () const override;
  String                    getTagName      () const;

  static String             getTagName

    ( const String&           rowItemName );

  virtual ItemSet*          getRowItems     () const = 0;

  virtual void              clearData       ();
  virtual void              clearAll        ()       = 0;
  virtual void              trimToSize      ();

  virtual IntColumn*        addIntColumn

    ( const String&           name )                 = 0;

  virtual IntColumn*        findIntColumn

    ( const String&           name )           const;

  IntColumn*                getIntColumn

    ( const String&           name )           const;

  virtual IntColArray       getIntColumns   () const = 0;

  virtual FloatColumn*      addFloatColumn

    ( const String&           name )                 = 0;

  virtual FloatColumn*      findFloatColumn

    ( const String&           name )           const;

  FloatColumn*              getFloatColumn

    ( const String&           name )           const;

  virtual FloatColArray     getFloatColumns () const = 0;

  virtual void              printTo

    ( PrintWriter&            out )            const;

  static void               printSome

    ( PrintWriter&            out,
      const Dictionary&       dbases,
      const String&           tag = "" );

  static void               printAll

    ( PrintWriter&            out,
      const Properties&       globdat );

  inline void               store

    ( const Properties&       globdat )        const;

  void                      store

    ( const String&           name,
      const Properties&       globdat )        const;

  static inline Ref<Self>   find

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Database>      find

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static inline Ref<Self>   get

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );

  static Ref<Database>      get

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );

  static Ref<Dictionary>    findFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Dictionary>    getFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  explicit                  Database

    ( const String&           name = "" );

  virtual                  ~Database        ();

};





//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class IntDBColumn
//=======================================================================

//-----------------------------------------------------------------------
//   findValue
//-----------------------------------------------------------------------


inline bool IntDBColumn::findValue

  ( idx_t&  value,
    idx_t   irow ) const

{
  return (getData( &value, 1, irow ) != 0);
}


//-----------------------------------------------------------------------
//   findValues
//-----------------------------------------------------------------------


inline idx_t IntDBColumn::findValues

  ( const IdxVector&  buf,
    idx_t             irow ) const

{
  return findValues ( buf, buf.size(), irow );
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


inline void IntDBColumn::getValues

  ( const IdxVector&  buf,
    idx_t             irow ) const

{
  getValues ( buf, buf.size(), irow );
}


template <int M, int N>

  inline void IntDBColumn::getValues

  ( Tuple<idx_t,M,N>&  buf,
    idx_t              irow ) const

{
  const idx_t  k = getData ( buf.addr(), M * N, irow );

  if ( k < (M * N) )
  {
    missingDataError_ ( irow, k, M * N );
  }
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void IntDBColumn::setValue

  ( idx_t  irow,
    idx_t  value )

{
  setData ( irow, &value, 1 );
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


inline void IntDBColumn::setValues

  ( idx_t             irow,
    const IdxVector&  buf )

{
  setValues ( irow, buf, buf.size() );
}


template <int M, int N>

  inline void IntDBColumn::setValues

  ( idx_t                    irow,
    const Tuple<idx_t,M,N>&  buf )

{
  setData ( irow, buf.addr(), M * N );
}


//=======================================================================
//   class FloatDBColumn
//=======================================================================

//-----------------------------------------------------------------------
//   findValue
//-----------------------------------------------------------------------


inline bool FloatDBColumn::findValue

  ( double&  value,
    idx_t    irow ) const

{
  return (getData( &value, 1, irow ) != 0);
}


//-----------------------------------------------------------------------
//   findValues
//-----------------------------------------------------------------------


inline idx_t FloatDBColumn::findValues

  ( const Vector&  buf,
    idx_t          irow ) const

{
  return findValues ( buf, buf.size(), irow );
}


//-----------------------------------------------------------------------
//   getValues
//-----------------------------------------------------------------------


inline void FloatDBColumn::getValues

  ( const Vector&  buf,
    idx_t          irow ) const

{
  getValues ( buf, buf.size(), irow );
}


template <int M, int N>

  inline void FloatDBColumn::getValues

  ( Tuple<double,M,N>&  buf,
    idx_t               irow ) const

{
  const idx_t  k = getData ( buf.addr(), M * N, irow );

  if ( k < (M * N) )
  {
    missingDataError_ ( irow, k, M * N );
  }
}


//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void FloatDBColumn::setValue

  ( idx_t   irow,
    double  value )

{
  setData ( irow, &value, 1 );
}


//-----------------------------------------------------------------------
//   setValues
//-----------------------------------------------------------------------


inline void FloatDBColumn::setValues

  ( idx_t          irow,
    const Vector&  buf )

{
  setValues ( irow, buf, buf.size() );
}


template <int M, int N>

  inline void FloatDBColumn::setValues

  ( idx_t                     irow,
    const Tuple<double,M,N>&  buf )

{
  setData ( irow, buf.addr(), M * N );
}


//=======================================================================
//   class Database
//=======================================================================

//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void Database::store ( const Properties& globdat ) const
{
  store ( myName_, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


inline Ref<Database>   Database::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  return find ( MAIN_DB, items, globdat );
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


inline Ref<Database>   Database::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat,
    const String&        context )

{
  return get ( MAIN_DB, items, globdat, context );
}


JIVE_END_PACKAGE( util )

#endif
