
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
#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/util/Flex.h>
#include <jem/util/Event.h>
#include <jem/util/Dictionary.h>
#include <jem/util/Properties.h>
#include <jem/util/FastAllocator.h>
#include <jem/xml/utilities.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/Tokenizer.h>
#include <jem/xml/ParserState.h>
#include <jive/util/private/NameList.h>
#include <jive/util/Globdat.h>
#include <jive/util/Reordering.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/Constraints.h>
#include <jive/util/ConstraintsParser.h>


JEM_DEFINE_CLASS( jive::util::ConstraintsParser );


JIVE_BEGIN_PACKAGE( util )


using jem::dynamicCast;
using jem::newInstance;
using jem::IllegalInputException;
using jem::util::Flex;
using jem::util::FastAllocator;
using jem::xml::ParseLog;
using jem::xml::Tokenizer;


//=======================================================================
//   class ConstraintsParser::Node_
//=======================================================================


class ConstraintsParser::Node_
{
 public:

  Node_*                  next;
  idx_t                   iitem;
  idx_t                   itype;
  double                  value;

};


//=======================================================================
//   class ConstraintsParser::Constraint_
//=======================================================================


class ConstraintsParser::Constraint_
{
 public:

  inline                  Constraint_   ();

  inline idx_t            itemCount     () const;

  inline idx_t            itemCount

    ( idx_t                 iop )          const;

  inline idx_t            operandCount  () const;
  inline void             clear         ();


 public:

  Flex<idx_t>             offsets;
  Flex<idx_t>             iitems;
  Flex<idx_t>             itypes;
  Flex<double>            values;

  bool                    rmask;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


inline ConstraintsParser::Constraint_::Constraint_ () :

  offsets ( 1 )

{
  offsets[0] = 0;
 rmask      = false;
}


//-----------------------------------------------------------------------
//   itemCount
//-----------------------------------------------------------------------


inline idx_t ConstraintsParser::Constraint_::itemCount () const
{
  return (offsets[1] - offsets[0]);
}


inline idx_t ConstraintsParser::Constraint_::itemCount

  ( idx_t iop ) const

{
  return (offsets[iop + 1] - offsets[iop]);
}


//-----------------------------------------------------------------------
//   operandCount
//-----------------------------------------------------------------------


inline idx_t ConstraintsParser::Constraint_::operandCount () const
{
  return itypes.size ();
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


inline void ConstraintsParser::Constraint_::clear ()
{
  offsets.resize ( 1 );
  offsets[0] = 0;

  iitems.clear ();
  itypes.clear ();
  values.clear ();

  rmask = false;
}


//=======================================================================
//   class ConstraintsParser::Data_
//=======================================================================


class ConstraintsParser::Data_ : public jem::Collectable
{
 public:

  typedef Data_             Self;


                            Data_

    ( const Ref<ItemSet>&     items,
      const Ref<Dictionary>&  groups );


  inline Node_*             newNode

    ( idx_t                   iitem,
      idx_t                   itype,
      double                  value );

  void                      clear  ();

  inline void               clearList

    ( Node_*&                 list );


 public:

  FastAllocator             allocator;

  Ref<ParserScope>          scope;
  Ref<ItemSet>              items;
  Ref<ItemMap>              itemMap;
  Ref<Dictionary>           itemGroups;
  NameList                  typeNames;
  Flex<Node_*>              constraints;
  Flex<bool>                rvalueMask;

  bool                      skipUnknown;


 protected:

  virtual                  ~Data_  ();


 private:

  void                      itemsResized_

    ( idx_t                   size );

  void                      itemsReordered_

    ( const Reordering&       reord );

};


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ConstraintsParser::Data_::Data_

  ( const Ref<ItemSet>&     iset,
    const Ref<Dictionary>&  groups ) :

    allocator   ( sizeof(Node_) ),
    items       (          iset ),
    itemGroups  (        groups ),
    skipUnknown (         false )

{
  using jem::util::connect;

  if ( items )
  {
    itemMap = items->getItemMap ();

    connect ( items->newSizeEvent,  this, & Self::itemsResized_   );
    connect ( items->newOrderEvent, this, & Self::itemsReordered_ );
  }
}


ConstraintsParser::Data_::~Data_ ()
{
  clear ();
}


//-----------------------------------------------------------------------
//   newNode
//-----------------------------------------------------------------------


inline ConstraintsParser::Node_* ConstraintsParser::Data_::newNode

  ( idx_t   iitem,
    idx_t   itype,
    double  value )

{
  Node_*  node = static_cast<Node_*> ( allocator.alloc() );

  node->next   = 0;
  node->value  = value;
  node->iitem  = iitem;
  node->itype  = itype;

  return node;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ConstraintsParser::Data_::clear ()
{
  const idx_t  n = constraints.size ();

  for ( idx_t icon = 0; icon < n; icon++ )
  {
    clearList ( constraints[icon] );
  }

  typeNames  .clear ();
  constraints.clear ();
  rvalueMask .clear ();
}


//-----------------------------------------------------------------------
//   clearList
//-----------------------------------------------------------------------


inline void ConstraintsParser::Data_::clearList ( Node_*& list )
{
  while ( list )
  {
    Node_*  node = list;

    list = list->next;

    allocator.dealloc ( node );
  }
}


//-----------------------------------------------------------------------
//   itemsResized_
//-----------------------------------------------------------------------


void ConstraintsParser::Data_::itemsResized_ ( idx_t itemCount )
{
  if ( itemCount == 0 )
  {
    clear ();
  }
}


//-----------------------------------------------------------------------
//   itemsReordered_
//-----------------------------------------------------------------------


void ConstraintsParser::Data_::itemsReordered_

  ( const Reordering&  reord )

{
  const idx_t* JEM_RESTRICT  fmap = reord.fmapPtr ();

  const idx_t  n = constraints.size ();

  Node_*       node;
  Node_*       con;

  idx_t        jitem;
  idx_t        icon;
  idx_t        jcon;


  for ( icon = jcon = 0; icon < n; icon++ )
  {
    con   = constraints[icon];
    jitem = fmap[con->iitem];

    if ( jitem < 0 )
    {
      clearList ( con );
    }
    else
    {
      con->iitem = jitem;
      node       = con->next;

      while ( node )
      {
        jitem = fmap[node->iitem];

        if ( jitem < 0 )
        {
          con->next = node->next;
          allocator.dealloc ( node );
        }
        else
        {
          con         = node;
          node->iitem = jitem;
        }

        node = con->next;
      }

      constraints[jcon] = constraints[icon];
      rvalueMask [jcon] = rvalueMask [icon];

      jcon++;
    }
  }

  constraints.resize ( jcon );
  rvalueMask .resize ( jcon );
}


//=======================================================================
//   class ConstraintsParser::Errors_
//=======================================================================


class ConstraintsParser::Errors_
{
 public:

  static void             wrongDofCount

    ( State&                state,
      idx_t                 actualCount,
      idx_t                 formalCount );

  static void             noSuchItem

    ( State&                state,
      const Data_&          data,
      idx_t                 itemID );

  static void             noSuchGroup

    ( State&                state,
      const Data_&          data,
      const String&         name );

  static void             illegalSlaveDof

    ( const String&         context,
      const Data_&          data,
      idx_t                 iitem,
      idx_t                 itype );

  static void             noSuchDof

    ( const String&         context,
      const Data_&          data,
      idx_t                 iitem,
      idx_t                 itype );

  static void             illegalItemSet

    ( const String&         context,
      const Data_&          data,
      const DofSpace&       dofs );

};


//-----------------------------------------------------------------------
//   wrongDofCount
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::wrongDofCount

  ( State&  state,
    idx_t   actualCount,
    idx_t   formalCount )

{
  String  msg;

  if ( formalCount == 1 )
  {
    msg =

      String::format (
        "wrong number (%d) of dofs selected; expected 1 dof",
        actualCount
      );
  }
  else
  {
    msg =

      String::format (
        "wrong number (%d) of dofs selected; expected 1 or %d dofs",
        actualCount,
        formalCount
      );
  }

  inputError ( state, msg );
}


//-----------------------------------------------------------------------
//   noSuchItem
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::noSuchItem

  ( State&         state,
    const Data_&   data,
    idx_t          itemID )

{
  const String  itemName = data.items->getItemName ();

  inputError (
    state,
    String::format (
      "undefined %s: %d", & itemName, itemID
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchGroup
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::noSuchGroup

  ( State&         state,
    const Data_&   data,
    const String&  name )

{
  const String  itemName = data.items->getItemName ();

  inputError (
    state,
    String::format (
      "undefined %s group: %s", itemName, name
    )
  );
}


//-----------------------------------------------------------------------
//   illegalSlaveDof
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::illegalSlaveDof

  ( const String&  context,
    const Data_&   data,
    idx_t          iitem,
    idx_t          itype )

{
  const String  dofType = data.typeNames.getName ( itype );
  const idx_t   itemID  = data.items->getItemID  ( iitem );

  throw IllegalInputException (
    context,
    String::format (
      "degree of freedom `%s[%d]\' marked as slave dof twice",
      dofType,
      itemID
    )
  );
}


//-----------------------------------------------------------------------
//   noSuchDof
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::noSuchDof

  ( const String&  context,
    const Data_&   data,
    idx_t            iitem,
    idx_t            itype )

{
  const String  dofType = data.typeNames.getName ( itype );
  const idx_t   itemID  = data.items->getItemID  ( iitem );

  throw IllegalInputException (
    context,
    String::format (
      "undefined degree of freedom: %s[%d]",
      dofType,
      itemID
    )
  );
}


//-----------------------------------------------------------------------
//   illegalItemSet
//-----------------------------------------------------------------------


void ConstraintsParser::Errors_::illegalItemSet

  ( const String&    context,
    const Data_&     data,
    const DofSpace&  dofs )

{
  const String  rightName = data.items->getName ();
  const String  wrongName = dofs.getItems()->getName ();

  throw jem::IllegalArgumentException (
    context,
    String::format (
      "dof space is associated with wrong item set: %s "
      "(expected item set `%s\')",
      wrongName,
      rightName
    )
  );
}


//=======================================================================
//   class ConstraintsParser::Utils_
//=======================================================================


class ConstraintsParser::Utils_ : public util::ParseUtils
{
 public:

  static void             parseConstraint

    ( State&                state,
      Constraint_&          con,
      Data_&                data );

  static void             parseOperand

    ( State&                state,
      Constraint_&          con,
      Data_&                data );

  static void             addRange

    ( State&                 state,
      Constraint_&           con,
      const Data_&           data,
      idx_t                  first,
      idx_t                  last,
      idx_t                  stride );

  static void             addGroup

    ( State&                state,
      Constraint_&          con,
      const Data_&          data,
      const String&         name );

  static void             addConstraint

    ( State&                state,
      Data_&                data,
      const Constraint_&    con );

  static void             addConstraints

    ( State&                state,
      Data_&                data,
      const Constraint_&    con );

  static IdxVector        getTypeMap

    ( const Data_&          data,
      const DofSpace&       dofs );

};


//-----------------------------------------------------------------------
//   parseConstraint
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::parseConstraint

  ( State&        state,
    Constraint_&  con,
    Data_&        data )

{
  Tokenizer&  input = * state.input;

  double      sign;
  double      value;
  int         token;


  con.clear ();
  con.values.pushBack ( 0.0 );

  parseOperand ( state, con, data );

  if ( input.nextToken() != '=' )
  {
    parseError ( state, "expected a `=\'" );
  }

  token = input.nextToken ();

  do
  {
    sign = 1.0;

    if      ( token == '+' )
    {
      token = input.nextToken ();
    }
    else if ( token == '-' )
    {
      sign  = -sign;
      token = input.nextToken ();
    }

    if ( token == Tokenizer::FLOAT_TOKEN ||
         token == Tokenizer::INTEGER_TOKEN )
    {
      value = sign * input.getFloat  ();
      token =        input.nextToken ();

      if ( token == '*' )
      {
        con.values.pushBack ( value );
        input.nextToken ();
        parseOperand ( state, con, data );

        token = input.nextToken ();
      }
      else
      {
        con.values[0] += value;
        con.rmask      = true;
      }
    }
    else
    {
      con.values.pushBack ( sign );
      parseOperand ( state, con, data );

      token = input.nextToken ();
    }
  }
  while ( token == '+' || token == '-' );

  if ( token != ';' )
  {
    parseError ( state, "expected a `;\'" );
  }

  if ( con.itemCount() == 1 )
  {
    addConstraint  ( state, data, con );
  }
  else
  {
    addConstraints ( state, data, con );
  }
}


//-----------------------------------------------------------------------
//   parseOperand
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::parseOperand

  ( State&        state,
    Constraint_&  con,
    Data_&        data )

{
  const Scope*  scope =   data.scope.get ();
  Tokenizer&    input = * state.input;

  String        typeName;

  int           token;
  idx_t         first, last, stride;
  idx_t         itype;
  idx_t         k;


  token = input.getToken ();

  if      ( token == Tokenizer::SYMBOL_TOKEN )
  {
    typeName = input.getSymbol ();
  }
  else if ( token == Tokenizer::STRING_TOKEN )
  {
    typeName = input.getString ();
  }
  else
  {
    parseError ( state );
  }

  itype = data.typeNames.findName ( typeName );

  if ( itype < 0 )
  {
    itype = data.typeNames.addName ( typeName );
  }

  if ( input.nextToken() != '[' )
  {
    parseError ( state, "expected a `[\'" );
  }

  token = input.nextToken ();

  if ( token == Tokenizer::INTEGER_TOKEN )
  {
    input.pushBackToken ();

    parseRange ( state,            first, last, stride );
    addRange   ( state, con, data, first, last, stride );

    token = input.getToken ();
  }
  else if ( token == Tokenizer::SYMBOL_TOKEN )
  {
    addGroup ( state, con, data, getSymbol( input, scope ) );

    token = input.nextToken ();
  }
  else if ( token == Tokenizer::STRING_TOKEN )
  {
    addGroup ( state, con, data, getString( input, scope ) );

    token = input.nextToken ();
  }
  else
  {
    parseError ( state );
  }

  if ( token != ']' )
  {
    parseError ( state, "expected a `]\'" );
  }

  k = con.itypes.size ();

  con.itypes .pushBack ( itype );
  con.offsets.pushBack ( con.iitems.size() );

  if ( con.itemCount(k) != con.itemCount() &&
       con.itemCount(k) != 1 )
  {
    Errors_::wrongDofCount (
      state,
      con.itemCount ( k ),
      con.itemCount ()
    );
  }
}


//-----------------------------------------------------------------------
//   addRange
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::addRange

  ( State&        state,
    Constraint_&  con,
    const Data_&  data,
    idx_t         first,
    idx_t         last,
    idx_t         stride )

{
  const Scope*  scope = data.scope.get ();

  idx_t         iitem;


  for ( ; first < last; first += stride )
  {
    if ( scope )
    {
      iitem = data.itemMap->findItem ( scope->expandID( first ) );
    }
    else
    {
      iitem = data.itemMap->findItem ( first );
    }

    if ( iitem >= 0 )
    {
      con.iitems.pushBack ( iitem );
    }
    else if ( ! data.skipUnknown )
    {
      Errors_::noSuchItem ( state, data, first );
    }
  }
}


//-----------------------------------------------------------------------
//   addGroup
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::addGroup

  ( State&         state,
    Constraint_&   con,
    const Data_&   data,
    const String&  name )

{
  Ref<ItemGroup>  group;


  if ( data.itemGroups )
  {
    group = dynamicCast<ItemGroup> ( data.itemGroups->get( name ) );

    if ( group && group->getItems() != data.items.get() )
    {
      group = nullptr;
    }
  }

  if ( ! group )
  {
    Errors_::noSuchGroup ( state, data, name );
  }

  IdxVector  iitems = group->getIndices ();

  con.iitems.pushBack ( iitems.begin(), iitems.end() );
}


//-----------------------------------------------------------------------
//   addConstraint
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::addConstraint

  ( State&              state,
    Data_&              data,
    const Constraint_&  con )

{
  const idx_t  opCount = con.operandCount ();

  Node_*       node;


  // Add the slave dof.

  node = data.newNode ( con.iitems[0],
                        con.itypes[0],
                        con.values[0] );

  data.constraints.pushBack ( node );
  data.rvalueMask .pushBack ( con.rmask );

  // Add the master dofs.

  for ( idx_t iop = 1; iop < opCount; iop++ )
  {
    node->next = data.newNode ( con.iitems[iop],
                                con.itypes[iop],
                                con.values[iop] );
    node = node->next;
  }
}


//-----------------------------------------------------------------------
//   addConstraints
//-----------------------------------------------------------------------


void ConstraintsParser::Utils_::addConstraints

  ( State&              state,
    Data_&              data,
    const Constraint_&  con )

{
  const idx_t   itemCount = con.itemCount         ();
  const idx_t   opCount   = con.operandCount      ();
  const idx_t   first     = data.constraints.size ();

  const idx_t*  iitems;
  const idx_t*  offsets;

  Node_*        node;
  Node_*        next;

  double        value;
  idx_t         itype;


  // Add the slave dofs.

  iitems  = con.iitems .addr ();
  offsets = con.offsets.addr ();
  itype   = con.itypes[0];
  value   = con.values[0];

  for ( idx_t i = 0; i < itemCount; i++ )
  {
    node = data.newNode ( iitems[i], itype, value );

    data.constraints.pushBack ( node );
    data.rvalueMask .pushBack ( con.rmask );
  }

  // Add the master dofs.

  for ( idx_t iop = opCount - 1; iop > 0; iop-- )
  {
    idx_t  i = offsets[iop];
    idx_t  n = offsets[iop + 1] - i;

    iitems   = con.iitems.addr ( i );
    itype    = con.itypes[iop];
    value    = con.values[iop];

    if      ( n == 1 )
    {
      for ( i = 0; i < itemCount; i++ )
      {
        next       = data.newNode ( iitems[0], itype, value );
        node       = data.constraints[i + first];
        next->next = node->next;
        node->next = next;
      }
    }
    else if ( n == itemCount )
    {
      for ( i = 0; i < itemCount; i++ )
      {
        next       = data.newNode ( iitems[i], itype, value );
        node       = data.constraints[i + first];
        next->next = node->next;
        node->next = next;
      }
    }
    else
    {
      throw jem::Error (
        JEM_FUNC,
        String::format (
          "illegal group size: %d; should be 1 or %d", n, itemCount
        )
      );
    }
  }
}


//-----------------------------------------------------------------------
//   getTypeMap
//-----------------------------------------------------------------------


IdxVector ConstraintsParser::Utils_::getTypeMap

  ( const Data_&     data,
    const DofSpace&  dofs )

{
  const idx_t  typeCount = data.typeNames.size ();

  IdxVector    map ( typeCount );


  for ( idx_t i = 0; i < typeCount; i++ )
  {
    idx_t  itype = dofs.findType ( data.typeNames.getName(i) );

    if ( itype < 0 )
    {
      String typeName = data.typeNames.getName ( i );

      throw IllegalInputException (
        CLASS_NAME,
        String::format (
          "unknown degree of freedom type: %s", typeName
        )
      );
    }

    map[i] = itype;
  }

  return map;
}


//=======================================================================
//   class ConstraintsParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ConstraintsParser::ConstraintsParser

  ( const Ref<ItemSet>&     items,
    const Ref<Dictionary>&  groups )

{
  JEM_PRECHECK2 ( items, "NULL ItemSet" );

  data_ = newInstance<Data_> ( items, groups );
}


ConstraintsParser::~ConstraintsParser ()
{}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void ConstraintsParser::parse ( State& state )
{
  Tokenizer&   input = * state.input;
  Data_&       d     = * data_;

  Constraint_  con;
  idx_t        conCount;
  int          token;


  d.constraints.reserve ( d.items->size() / 16 );
  d.rvalueMask .reserve ( d.constraints.size() );

  conCount = 0;
  token    = input.nextToken ();

  while ( token == Tokenizer::SYMBOL_TOKEN ||
          token == Tokenizer::STRING_TOKEN )
  {
    try
    {
      Utils_::parseConstraint ( state, con, d );
    }
    catch ( const ItemIDException& ex )
    {
      Utils_::itemIDError ( state, *d.items, ex );
    }

    conCount++;

    if ( conCount % 1000_idx == 0 )
    {
      state.log->logEvent (
        state,
        ParseLog::PROGRESS,
        String::format (
          "number of contraints read : %d", conCount
        )
      );
    }

    token = input.nextToken ();
  }

  input.pushBackToken ();

  JEM_ASSERT ( d.constraints.size() == d.rvalueMask.size() );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void ConstraintsParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Data_&  d = * data_;


  if      ( action == ParserActions::NEW_SCOPE )
  {
    params.get ( d.scope, ParserParams::SCOPE );

    if ( d.scope->isGlobal() )
    {
      d.scope = nullptr;
    }
  }
  else if ( action == ParserActions::SET_OPTIONS )
  {
    params.find ( d.skipUnknown, ParserOptions::SKIP_UNKNOWN );
  }
  else if ( action == ParserActions::RESET_OPTIONS )
  {
    d.skipUnknown = false;
  }
}


//-----------------------------------------------------------------------
//   getTagName
//-----------------------------------------------------------------------


String ConstraintsParser::getTagName () const
{
  String itemName = data_->items->getItemName ();

  return jem::xml::makeTag ( itemName ) + "Constraints";
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ConstraintsParser::clear ()
{
  data_->clear ();
}


//-----------------------------------------------------------------------
//   slaveDofCount
//-----------------------------------------------------------------------


idx_t ConstraintsParser::slaveDofCount () const
{
  return data_->constraints.size ();
}


//-----------------------------------------------------------------------
//   getItems
//-----------------------------------------------------------------------


ItemSet* ConstraintsParser::getItems () const
{
  return data_->items.get ();
}


//-----------------------------------------------------------------------
//   addDofsTo
//-----------------------------------------------------------------------


void ConstraintsParser::addDofsTo ( XDofSpace& dofs ) const
{
  const Data_&  d         = * data_;

  const idx_t   conCount  = d.constraints.size ();
  const idx_t   typeCount = d.typeNames  .size ();

  IdxVector     typeMap   ( typeCount );

  Node_*        con;
  Node_*        node;


  if ( d.items != dofs.getItems() )
  {
    Errors_::illegalItemSet ( JEM_FUNC, d, dofs );
  }

  for ( idx_t i = 0; i < typeCount; i++ )
  {
    typeMap[i] = dofs.addType ( d.typeNames.getName(i) );
  }

  for ( idx_t icon = 0; icon < conCount; icon++ )
  {
    con = d.constraints[icon];

    dofs.addDof ( con->iitem, typeMap[con->itype] );

    for ( node = con->next; node; node = node->next )
    {
      dofs.addDof ( node->iitem, typeMap[node->itype] );
    }
  }
}


//-----------------------------------------------------------------------
//   getConstraints
//-----------------------------------------------------------------------


void ConstraintsParser::getConstraints

  ( Constraints&   cons,
    const String&  context ) const

{
  const Data_&     d        = * data_;
  const DofSpace&  dofs     = * cons.getDofSpace ();

  const idx_t      conCount = d.constraints.size ();
  const idx_t      dofCount = dofs.dofCount      ();

  IdxVector        typeMap  = Utils_::getTypeMap ( d, dofs );

  BoolVector       mask     ( dofCount );
  Vector           accu     ( dofCount );
  IdxVector        jdofs    ( dofCount );
  Vector           coeffs   ( dofCount );


  if ( d.items != dofs.getItems() )
  {
    Errors_::illegalItemSet ( context, d, dofs );
  }

  mask = false;
  accu = 0.0;

  for ( idx_t icon = 0; icon < conCount; icon++ )
  {
    Node_*  con  = d.constraints[icon];
    idx_t   idof = dofs.findDofIndex ( con->iitem, typeMap[con->itype] );

    if ( idof < 0 )
    {
      Errors_::noSuchDof ( context, d, con->iitem, con->itype );
    }

    if ( cons.isSlaveDof(idof) )
    {
      Errors_::illegalSlaveDof ( context, d,
                                 con->iitem, con->itype );
    }

    bool    rmask = d.rvalueMask[icon];
    Node_*  node  = con->next;

    if ( ! node )
    {
      if ( rmask )
      {
        cons.addConstraint ( idof, con->value );
      }
      else
      {
        cons.addConstraint ( idof );
      }

      continue;
    }

    idx_t  last = 0;

    do
    {
      idx_t  jdof = dofs.findDofIndex ( node->iitem,
                                        typeMap[node->itype] );

      if ( jdof < 0 )
      {
        Errors_::noSuchDof ( context, d,
                             node->iitem, node->itype );
      }

      if ( ! mask[jdof] )
      {
        jdofs[last++] = jdof;
        mask [jdof]   = true;
      }

      accu[jdof] += node->value;
      node        = node->next;
    }
    while ( node );

    for ( idx_t i = 0; i < last; i++ )
    {
      idx_t  jdof = jdofs[i];

      coeffs[i]   = accu[jdof];
      mask[jdof]  = false;
      accu[jdof]  = 0.0;
    }

    try
    {
      if ( last == 1 )
      {
        if ( rmask )
        {
          cons.addConstraint ( idof, con->value,
                               jdofs[0], coeffs[0] );
        }
        else
        {
          cons.addConstraint ( idof, jdofs[0], coeffs[0] );
        }
      }
      else
      {
        if ( rmask )
        {
          cons.addConstraint ( idof,
                               con->value,
                               jdofs [slice(BEGIN,last)],
                               coeffs[slice(BEGIN,last)] );
        }
        else
        {
          cons.addConstraint ( idof,
                               jdofs [slice(BEGIN,last)],
                               coeffs[slice(BEGIN,last)] );
        }
      }
    }
    catch ( IllegalInputException& ex )
    {
      ex.setContext ( context );
      throw;
    }
  }
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------


void ConstraintsParser::store ( const Properties& globdat ) const
{
  Ref<ItemSet>  items = data_->items;
  Ref<Self>     self  = const_cast<Self*> ( this );

  Globdat::storeFor ( Globdat::CONSTRAINTS_PARSER,
                      self, items, globdat );
}


//-----------------------------------------------------------------------
//   find
//-----------------------------------------------------------------------


Ref<ConstraintsParser> ConstraintsParser::find

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self>  self;

  Globdat::findFor ( self,  Globdat::CONSTRAINTS_PARSER,
                     items, globdat );

  return self;
}


//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------


Ref<ConstraintsParser> ConstraintsParser::get

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Properties  itemdat = Globdat::getDataFor ( items, globdat );
  Ref<Self>   self;


  if ( ! itemdat.find( self, Globdat::CONSTRAINTS_PARSER ) )
  {
    self = newInstance<Self> (
      items,
      ItemGroup::findFor ( items, globdat )
    );

    itemdat.set ( Globdat::CONSTRAINTS_PARSER, self );
  }

  return self;
}


//-----------------------------------------------------------------------
//   extract
//-----------------------------------------------------------------------


Ref<ConstraintsParser> ConstraintsParser::extract

  ( const Ref<ItemSet>&  items,
    const Properties&    globdat )

{
  Ref<Self> self;

  Globdat::extractFor ( self,  Globdat::CONSTRAINTS_PARSER,
                        items, globdat );

  return self;
}


JIVE_END_PACKAGE( util )
