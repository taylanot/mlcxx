
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

#ifndef JIVE_UTIL_TABLE_H
#define JIVE_UTIL_TABLE_H

#include <jem/base/NamedObject.h>
#include <jive/Array.h>
#include <jive/SparseMatrix.h>
#include <jive/util/import.h>
#include <jive/util/EventSource.h>


JIVE_BEGIN_PACKAGE( util )


class ItemSet;


//-----------------------------------------------------------------------
//   class Table
//-----------------------------------------------------------------------


class Table : public NamedObject,
              public EventSource
{
 public:

  JEM_DECLARE_CLASS       ( Table, NamedObject );

  Event < Self& >           newValuesEvent;
  Event < Self& >           newStructEvent;


  virtual String            getContext      () const override;
  virtual idx_t             size            () const = 0;
  idx_t                     rowCount        () const;
  virtual idx_t             columnCount     () const = 0;
  inline  bool              isDense         () const;

  virtual idx_t             findColumn

    ( const String&           name )           const;

  inline idx_t              getColumnIndex

    ( const String&           name )           const;

  virtual String            getColumnName

    ( idx_t                   jcol )           const = 0;

  virtual StringVector      getColumnNames  () const;
  virtual ItemSet*          getRowItems     () const = 0;
  virtual SparseMatrix      toMatrix        () const = 0;

  virtual idx_t             getData

    ( double*                 buf,
      const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )         const = 0;

  virtual idx_t             getAllData

    ( double*                 buf,
      idx_t                   jst )            const = 0;

  inline bool               findValue

    ( double&                 value,
      idx_t                   irow,
      idx_t                   jcol )           const;

  inline double             getValue

    ( idx_t                   irow,
      idx_t                   jcol )           const;

  idx_t                     findBlock

    ( const Matrix&           block,
      const IdxVector&        irows,
      const IdxVector&        jcols )          const;

  void                      getBlock

    ( const Matrix&           block,
      const IdxVector&        irows,
      const IdxVector&        jcols )          const;

  idx_t                     findRowValues

    ( const Vector&           rvals,
      idx_t                   irow,
      const IdxVector&        jcols )          const;

  void                      getRowValues

    ( const Vector&           rvals,
      idx_t                   irow,
      const IdxVector&        jcols )          const;

  idx_t                     findColValues

    ( const Vector&           cvals,
      const IdxVector&        irows,
      idx_t                   jcol )           const;

  void                      getColValues

    ( const Vector&           cvals,
      const IdxVector&        irows,
      idx_t                   jcol )           const;

  idx_t                     findAllValues

    ( const Matrix&           buf )            const;

  void                      getAllValues

    ( const Matrix&           buf )            const;

  String                    getTagName      () const;

  static String             getTagName

    ( const String&           rowItemName );

  virtual void              printTo

    ( PrintWriter&            out )            const;

  static void               printSome

    ( PrintWriter&            out,
      const Dictionary&       tables,
      const String&           tag = "" );

  static void               printAll

    ( PrintWriter&            out,
      const Properties&       globdat );

  static void               printMatrix

    ( PrintWriter&            out,
      const SparseMatrix&     mat,
      const StringVector&     colNames,
      const IdxVector&        rowIDs  );

  static void               printSection

    ( PrintWriter&            out,
      const Matrix&           mat,
      const StringVector&     colNames,
      const IdxVector&        rowIDs,
      const IdxVector&        rowPerm );

  inline void               store

    ( const Properties&       globdat )        const;

  void                      store

    ( const String&           name,
      const Properties&       globdat )        const;

  static Ref<Table>         find

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Table>         get

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat,
      const String&           context );

  static Ref<Dict>          findFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<Dict>          getFor

    ( const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  explicit                  Table

    ( const String&           name = "" );

  virtual                  ~Table           ();


 private:

  void                      noSuchColumnError_

    ( const String&           name )           const;

  void                      noSuchValueError_

    ( idx_t                   irow,
      idx_t                   jcol )           const;

  void                      missingValueError_

    ( const IdxVector&        irows,
      const IdxVector&        jcols )          const;

  void                      missingValueError_

    ( idx_t                   irow,
      const IdxVector&        jcols )          const;

  void                      missingValueError_

    ( const IdxVector&        irows,
      idx_t                   jcol )           const;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   isDense
//-----------------------------------------------------------------------


inline bool Table::isDense () const
{
  const double  fillFrac =

    (double) size() / ((double) rowCount() *
                       (double) columnCount() + 0.1);

  return (fillFrac > 0.5);
}


//-----------------------------------------------------------------------
//   getColumnIndex
//-----------------------------------------------------------------------


inline idx_t Table::getColumnIndex

  ( const String&  name ) const

{
  idx_t  jcol = findColumn ( name );

  if ( jcol < 0 )
  {
    noSuchColumnError_ ( name );
  }

  return jcol;
}


//-----------------------------------------------------------------------
//   findValue
//-----------------------------------------------------------------------


inline bool Table::findValue

  ( double&  val,
    idx_t    irow,
    idx_t    jcol ) const

{
  return (getData( &val, &irow, 1, &jcol, 1 ) != 0);
}


//-----------------------------------------------------------------------
//   getValue
//-----------------------------------------------------------------------


inline double Table::getValue

  ( idx_t  irow,
    idx_t  jcol ) const

{
  double  val;

  if ( ! getData( &val, &irow, 1, &jcol, 1 ) )
  {
    noSuchValueError_ ( irow, jcol );
  }

  return val;
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


inline void Table::store ( const Properties& globdat ) const
{
  store ( myName_, globdat );
}


JIVE_END_PACKAGE( util )

#endif
