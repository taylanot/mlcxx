
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

#ifndef JIVE_UTIL_XTABLE_H
#define JIVE_UTIL_XTABLE_H

#include <jive/util/Table.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   class XTable
//-----------------------------------------------------------------------


class XTable : public Table
{
 public:

  JEM_DECLARE_CLASS       ( XTable, Table );


  virtual void              clearData   ()       = 0;
  virtual void              clearAll    ()       = 0;

  virtual void              reserve

    ( idx_t                   nzCount );

  virtual void              trimToSize  ();

  virtual idx_t             addColumn

    ( const String&           name )             = 0;

  virtual IdxVector         addColumns

    ( const StringVector&     names );

  virtual void              setData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )           = 0;

  virtual void              addData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount,
      const double*           values )           = 0;

  virtual idx_t             eraseData

    ( const idx_t*            irows,
      idx_t                   icount,
      const idx_t*            jcols,
      idx_t                   jcount )           = 0;

  virtual void              scaleRows

    ( const Vector&           scale );

  inline void               setValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline void               addValue

    ( idx_t                   irow,
      idx_t                   jcol,
      double                  value );

  inline bool               eraseValue

    ( idx_t                   irow,
      idx_t                   jcol );

  void                      setBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols,
      const Matrix&           block );

  void                      addBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols,
      const Matrix&           block );

  idx_t                     eraseBlock

    ( const IdxVector&        irows,
      const IdxVector&        jcols );

  void                      setRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols,
      const Vector&           rvals );

  void                      addRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols,
      const Vector&           rvals );

  idx_t                     eraseRowValues

    ( idx_t                   irow,
      const IdxVector&        jcols );

  void                      setColValues

    ( const IdxVector&        irows,
      idx_t                   jcol,
      const Vector&           cvals );

  void                      addColValues

    ( const IdxVector&        irows,
      idx_t                   jcol,
      const Vector&           cvals );

  idx_t                     eraseColValues

    ( const IdxVector&        irows,
      idx_t                   jcol );

  static Ref<XTable>        find

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );

  static Ref<XTable>        get

    ( const String&           name,
      const Ref<ItemSet>&     items,
      const Properties&       globdat );


 protected:

  explicit                  XTable

    ( const String&           name = "" );

  virtual                  ~XTable      ();

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   setValue
//-----------------------------------------------------------------------


inline void XTable::setValue

  ( idx_t   irow,
    idx_t   jcol,
    double  val )

{
  setData ( &irow, 1, &jcol, 1, &val );
}


//-----------------------------------------------------------------------
//   addValue
//-----------------------------------------------------------------------


inline void XTable::addValue

  ( idx_t   irow,
    idx_t    jcol,
    double val )

{
  addData ( &irow, 1, &jcol, 1, &val );
}


//-----------------------------------------------------------------------
//   eraseValue
//-----------------------------------------------------------------------


inline bool XTable::eraseValue

  ( idx_t  irow,
    idx_t  jcol )

{
  return (eraseData( &irow, 1, &jcol, 1 ) != 0);
}


JIVE_END_PACKAGE( util )

#endif
