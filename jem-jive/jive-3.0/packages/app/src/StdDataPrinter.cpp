
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


#include <jem/base/assert.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/mp/GatherPrinter.h>
#include <jem/util/Event.h>
#include <jem/xml/print.h>
#include <jive/util/Table.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/TableUtils.h>
#include <jive/app/Globdat.h>
#include <jive/app/StdDataPrinter.h>


JEM_DEFINE_CLASS( jive::app::StdDataPrinter );


JIVE_BEGIN_PACKAGE( app )


using jem::iarray;
using jem::newInstance;
using jive::util::ItemMap;
using jive::util::ItemSet;
using jive::util::TableUtils;


//=======================================================================
//   class FastTablePrinter
//=======================================================================


class FastTablePrinter : public Object
{
 public:

  typedef
    FastTablePrinter        Self;
  typedef jem::
    mp::GatherPrinter       Output;


  explicit inline           FastTablePrinter

    ( const Ref<ItemSet>&     items );

  inline void               printTable

    ( Output&                 out,
      const Table&            table );


 protected:

  virtual                  ~FastTablePrinter  ();

  void                      update_

    ( ItemSet&                items );

  void                      invalidate_       ();


 protected:

  IdxVector                 rowIDs_;
  IdxVector                 rowPerm_;

  bool                      updated_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline FastTablePrinter::FastTablePrinter

  ( const Ref<ItemSet>& items )

{
  using jem::util::connect;

  updated_ = false;

  connect ( items->newSizeEvent,  this, & Self::invalidate_ );
  connect ( items->newOrderEvent, this, & Self::invalidate_ );
}


FastTablePrinter::~FastTablePrinter ()
{}


//-----------------------------------------------------------------------
//   printTable
//-----------------------------------------------------------------------


inline void FastTablePrinter::printTable

  ( Output&       out,
    const Table&  table )

{
  const idx_t  rowCount = table.rowCount    ();
  const idx_t  colCount = table.columnCount ();

  Matrix       buf ( rowCount, colCount );


  JEM_ASSERT ( rowCount * colCount == table.size() );

  if ( ! updated_ )
  {
    update_ ( * table.getRowItems() );
  }

  table.getAllData    ( buf.addr(), buf.stride(1) );

  Table::printSection ( out, buf,
                        table.getColumnNames (),
                        rowIDs_, rowPerm_ );
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void FastTablePrinter::update_ ( ItemSet& items )
{
  const ItemMap&  imap     = * items.getItemMap ();

  const idx_t     rowCount = items.size ();


  rowIDs_ .resize   ( rowCount );
  rowPerm_.resize   ( rowCount );

  rowPerm_ = iarray ( rowCount );

  imap.getItemIDs   ( rowIDs_,  rowPerm_ );
  sort              ( rowPerm_, rowIDs_  );

  items.resetEvents ();

  updated_ = true;
}


//-----------------------------------------------------------------------
//   invalidate_
//-----------------------------------------------------------------------


void FastTablePrinter::invalidate_ ()
{
  updated_ = false;
}


//=======================================================================
//   class FastVectorPrinter
//=======================================================================


class FastVectorPrinter : public FastTablePrinter
{
 public:

  typedef
    FastVectorPrinter       Self;
  typedef
    FastTablePrinter        Super;


  explicit inline           FastVectorPrinter

    ( const Ref<DofSpace>&    dofs );

  inline void               printVector

    ( Output&                 out,
      const Vector&           vec );


 protected:

  virtual                  ~FastVectorPrinter ();

  void                      update_           ();


 protected:

  Ref<DofSpace>             dofs_;
  IdxVector                 dofPerm_;

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


inline FastVectorPrinter::FastVectorPrinter

  ( const Ref<DofSpace>&  dofs ) :

    Super ( dofs->getItems() ),
    dofs_ ( dofs )

{
  using jem::util::connect;

  connect ( dofs->newSizeEvent,  this, & Self::invalidate_ );
  connect ( dofs->newOrderEvent, this, & Self::invalidate_ );
}


FastVectorPrinter::~FastVectorPrinter ()
{}


//-----------------------------------------------------------------------
//   printVector
//-----------------------------------------------------------------------


inline void FastVectorPrinter::printVector

  ( Output&        out,
    const Vector&  vec )

{
  const idx_t  dofCount = vec   .size      ();
  const idx_t  rowCount = dofs_->itemCount ();
  const idx_t  colCount = dofs_->typeCount ();

  Vector       buf ( dofCount );


  JEM_ASSERT ( rowCount * colCount == dofCount );

  if ( ! updated_ )
  {
    update_ ();
  }

  buf = vec[dofPerm_];

  Table::printSection (
    out, reshape ( buf, jem::shape( rowCount, colCount ) ),
    dofs_->getTypeNames (),
    rowIDs_, rowPerm_
  );
}


//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------


void FastVectorPrinter::update_ ()
{
  const idx_t  dofCount  = dofs_->dofCount  ();
  const idx_t  itemCount = dofs_->itemCount ();
  const idx_t  typeCount = dofs_->typeCount ();

  IdxVector    iitems ( iarray( itemCount ) );
  IdxVector    itypes ( iarray( typeCount ) );

  IdxVector    idofs  ( dofCount );

  idx_t        i, j, k;


  Super::update_       ( * dofs_->getItems() );
  dofs_->getDofIndices ( idofs, iitems, itypes );

  // Store the dof indices in column-major order

  dofPerm_.resize ( dofCount );

  k = 0;

  for ( j = 0; j < typeCount; j++ )
  {
    for ( i = 0; i < itemCount; i++ )
    {
      dofPerm_[k++] = idofs[i * typeCount + j];
    }
  }

  dofs_->resetEvents ();
}


//=======================================================================
//   class StdDataPrinter::Utils_
//=======================================================================


class StdDataPrinter::Utils_
{
 public:

  static inline String    tableType

    ( idx_t                 size,
      idx_t                 rowCount,
      idx_t                 colCount );

  static inline void      beginTable

    ( Output&               out,
      const String&         tag,
      const String&         label,
      const String&         type );

  static inline void      endTable

    ( Output&               out,
      const String&         tag );

  static void             printFast

    ( Output&               out,
      const Table&          table,
      const Properties&     globdat );

  static void             printFast

    ( Output&               out,
      const Vector&         vec,
      const Ref<DofSpace>&  dofs,
      const Properties&     globdat );

};


//-----------------------------------------------------------------------
//   tableType
//-----------------------------------------------------------------------


inline String StdDataPrinter::Utils_::tableType

  ( idx_t  size,
    idx_t  rowCount,
    idx_t  colCount )

{
  if ( 2 * size > (rowCount * colCount) )
  {
    return "Dense";
  }
  else
  {
    return "Sparse";
  }
}


//-----------------------------------------------------------------------
//   beginTable
//-----------------------------------------------------------------------


inline void StdDataPrinter::Utils_::beginTable

  ( Output&        out,
    const String&  tag,
    const String&  label,
    const String&  type )

{
  using jem::io::blankLine;
  using jem::xml::beginStartTag;
  using jem::xml::attribute;
  using jem::xml::endStartTag;

  print ( out, beginStartTag( tag ), attribute( "name", label ),
          attribute( "type", type  ), endStartTag, blankLine );

  out.beginSession ();
}


//-----------------------------------------------------------------------
//   endTable
//-----------------------------------------------------------------------


inline void StdDataPrinter::Utils_::endTable

  ( Output&        out,
    const String&  tag )

{
  using jem::io::blankLine;
  using jem::xml::endTag;

  out.endSession ();

  print ( out, blankLine, endTag( tag ), blankLine );
}


//-----------------------------------------------------------------------
//   printFast (Table)
//-----------------------------------------------------------------------


void StdDataPrinter::Utils_::printFast

  ( Output&            out,
    const Table&       table,
    const Properties&  globdat )

{
  Ref<ItemSet>           items = table.getRowItems ();
  Ref<FastTablePrinter>  prn;


  Globdat::findFor ( prn, Globdat::FAST_PRINTER, items, globdat );

  if ( ! prn )
  {
    prn = newInstance<FastTablePrinter> ( items );

    Globdat::storeFor ( Globdat::FAST_PRINTER,
                        prn, items, globdat );
  }

  prn->printTable ( out, table );
}


//-----------------------------------------------------------------------
//   printFast (Vector)
//-----------------------------------------------------------------------


void StdDataPrinter::Utils_::printFast

  ( Output&               out,
    const Vector&         vec,
    const Ref<DofSpace>&  dofs,
    const Properties&     globdat )

{
  Ref<FastVectorPrinter>  prn;


  Globdat::findFor ( prn, Globdat::FAST_PRINTER, dofs, globdat );

  if ( ! prn )
  {
    Ref<ItemSet>  items = dofs->getItems ();

    prn = newInstance<FastVectorPrinter> ( dofs );

    Globdat::storeFor ( Globdat::FAST_PRINTER,
                        prn, dofs,  globdat );
    Globdat::storeFor ( Globdat::FAST_PRINTER,
                        prn, items, globdat );
  }

  prn->printVector ( out, vec );
}


//=======================================================================
//   class StdDataPrinter
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdDataPrinter::StdDataPrinter ( const String& name ) :

  Super ( name )

{}


StdDataPrinter::~StdDataPrinter ()
{}


//-----------------------------------------------------------------------
//   printTable
//-----------------------------------------------------------------------


void  StdDataPrinter::printTable

  ( Output&            out,
    const String&      label,
    const Table&       table,
    const Properties&  globdat )

{
  const idx_t  size     = table.size        ();
  const idx_t  rowCount = table.rowCount    ();
  const idx_t  colCount = table.columnCount ();

  String       tag  = table.getTagName  ();
  String       type = Utils_::tableType ( size, rowCount, colCount );


  Utils_::beginTable  ( out, tag, label, type );

  if ( size == (rowCount * colCount) )
  {
    Utils_::printFast ( out, table, globdat );
  }
  else
  {
    table.printTo     ( out );
  }

  Utils_::endTable    ( out, tag );
}


//-----------------------------------------------------------------------
//   printVector
//-----------------------------------------------------------------------


void StdDataPrinter::printVector

  ( Output&            out,
    const String&      label,
    const Vector&      vec,
    const DofSpace&    dofs,
    const Properties&  globdat )

{
  const ItemSet*  items = dofs.getItems  ();

  const idx_t  size     = vec   .size    ();
  const idx_t  rowCount = items->size    ();
  const idx_t  colCount = dofs.typeCount ();

  String       tag  = Table::getTagName ( items->getItemName() );
  String       type = Utils_::tableType ( size, rowCount, colCount );


  Utils_::beginTable ( out, tag, label, type );

  if ( size == (rowCount * colCount) )
  {
    Ref<DofSpace>  dofsRef = const_cast<DofSpace*> ( &dofs );

    Utils_::printFast ( out, vec, dofsRef, globdat );
  }
  else
  {
    TableUtils::printVector ( out, vec, dofs );
  }

  Utils_::endTable ( out, tag );
}


//-----------------------------------------------------------------------
//   printSparseVector
//-----------------------------------------------------------------------


void StdDataPrinter::printSparseVector

  ( Output&            out,
    const String&      label,
    const Vector&      vec,
    const IdxVector&   idofs,
    const DofSpace&    dofs,
    const Properties&  globdat )

{
  const ItemSet*  items = dofs.getItems ();

  String  tag  = Table::getTagName ( items->getItemName() );
  String  type = "Sparse";


  Utils_::beginTable      ( out, tag, label, type );
  TableUtils::printVector ( out, vec, idofs, dofs );
  Utils_::endTable        ( out, tag );
}


JIVE_END_PACKAGE( app )
