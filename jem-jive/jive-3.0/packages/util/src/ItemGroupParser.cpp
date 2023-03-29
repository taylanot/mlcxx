
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
#include <jem/base/Once.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/select.h>
#include <jem/util/Flex.h>
#include <jem/util/ObjFlex.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jem/xml/ParseLog.h>
#include <jem/xml/ParserState.h>
#include <jive/Array.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ItemMap.h>
#include <jive/util/ItemGroup.h>
#include <jive/util/XItemGroup.h>
#include <jive/util/ItemIDException.h>
#include <jive/util/ParseUtils.h>
#include <jive/util/ParserActions.h>
#include <jive/util/ItemGroupParser.h>


JEM_DEFINE_CLASS( jive::util::ItemGroupParser );


JIVE_BEGIN_PACKAGE( util )


using jem::staticCast;
using jem::newInstance;
using jem::util::Flex;
using jem::util::ObjFlex;


//=======================================================================
//   class ItemGroupParser::Operator_
//=======================================================================


class ItemGroupParser::Operator_
{
 public:

  enum                    Assoc
  {
                            LEFT_TO_RIGHT,
                            RIGHT_TO_LEFT
  };


  typedef IdxVector     (*OpFunc1)

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      rhs );

  typedef IdxVector     (*OpFunc2)

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      lhs,
      const IdxVector&      rhs );


  explicit inline         Operator_

    ( OpFunc1               func,
      const char*           name,
      short                 rank,
      Assoc                 assoc );

  explicit inline         Operator_

    ( OpFunc2               func,
      const char*           name,
      short                 rank,
      Assoc                 assoc );

  inline bool             isUnary  () const;
  inline bool             isBinary () const;

  static IdxVector        orFunc

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      lhs,
      const IdxVector&      rhs );

  static IdxVector        andFunc

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      lhs,
      const IdxVector&      rhs );

  static IdxVector        andNotFunc

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      lhs,
      const IdxVector&      rhs );

  static IdxVector        notFunc

    ( const BoolVector&     mask,
      const IdxVector&      ibuf,
      const IdxVector&      rhs );


 public:

  OpFunc1                 func1;
  OpFunc2                 func2;
  const char*             name;
  const short             rank;
  const Assoc             assoc;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline ItemGroupParser::Operator_::Operator_

  ( OpFunc1      func,
    const char*  name,
    short        prec,
    Assoc        assoc ) :

    func1 (  func ),
    func2 (     0 ),
    name  (  name ),
    rank  (  prec ),
    assoc ( assoc )

{}


inline ItemGroupParser::Operator_::Operator_

  ( OpFunc2      func,
    const char*  name,
    short        prec,
    Assoc        assoc ) :

    func1 (     0 ),
    func2 (  func ),
    name  (  name ),
    rank  (  prec ),
    assoc ( assoc )

{}


//-----------------------------------------------------------------------
//   isUnary
//-----------------------------------------------------------------------


inline bool ItemGroupParser::Operator_::isUnary () const
{
  return (func1 != 0);
}


//-----------------------------------------------------------------------
//   isBinary
//-----------------------------------------------------------------------


inline bool ItemGroupParser::Operator_::isBinary () const
{
  return (func2 != 0);
}


//-----------------------------------------------------------------------
//   orFunc
//-----------------------------------------------------------------------


IdxVector ItemGroupParser::Operator_::orFunc

  ( const BoolVector&  mask,
    const IdxVector&   ibuf,
    const IdxVector&   lhs,
    const IdxVector&   rhs )

{
  idx_t  iitem;
  idx_t  i, k, n;


  n = 0;
  k = lhs.size ();

  for ( i = 0; i < k; i++ )
  {
    iitem = lhs[i];

    if ( ! mask[iitem] )
    {
      ibuf[n++]   = iitem;
      mask[iitem] = true;
    }
  }

  k = rhs.size ();

  for ( i = 0; i < k; i++ )
  {
    iitem = rhs[i];

    if ( ! mask[iitem] )
    {
      ibuf[n++]   = iitem;
      mask[iitem] = true;
    }
  }

  IdxVector  result ( ibuf[slice(BEGIN,n)].clone() );

  mask[result] = false;

  return result;
}


//-----------------------------------------------------------------------
//   andFunc
//-----------------------------------------------------------------------


IdxVector ItemGroupParser::Operator_::andFunc

  ( const BoolVector&  mask,
    const IdxVector&   ibuf,
    const IdxVector&   lhs,
    const IdxVector&   rhs )

{
  idx_t  iitem;
  idx_t  i, k, n;


  mask[lhs] = true;

  n = 0;
  k = rhs.size ();

  for ( i = 0; i < k; i++ )
  {
    iitem = rhs[i];

    if ( mask[iitem] )
    {
      ibuf[n++]   = iitem;
      mask[iitem] = false;
    }
  }

  mask[lhs] = false;

  return ibuf[slice(BEGIN,n)].clone ();
}


//-----------------------------------------------------------------------
//   andNotFunc
//-----------------------------------------------------------------------


IdxVector ItemGroupParser::Operator_::andNotFunc

  ( const BoolVector&  mask,
    const IdxVector&   ibuf,
    const IdxVector&   lhs,
    const IdxVector&   rhs )

{
  idx_t  iitem;
  idx_t  i, k, n;


  mask[lhs] = true;
  mask[rhs] = false;

  n = 0;
  k = lhs.size ();

  for ( i = 0; i < k; i++ )
  {
    iitem = lhs[i];

    if ( mask[iitem] )
    {
      ibuf[n++]   = iitem;
      mask[iitem] = false;
    }
  }

  return ibuf[slice(BEGIN,n)].clone ();
}


//-----------------------------------------------------------------------
//   notFunc
//-----------------------------------------------------------------------


IdxVector ItemGroupParser::Operator_::notFunc

  ( const BoolVector&  mask,
    const IdxVector&   ibuf,
    const IdxVector&   rhs )

{
  idx_t  i, k, n;


  mask[rhs] = true;

  n = 0;
  k = mask.size ();

  for ( i = 0; i < k; i++ )
  {
    if ( ! mask[i] )
    {
      ibuf[n++] = i;
    }

    mask[i] = false;
  }

  return ibuf[slice(BEGIN,n)].clone ();
}


//=======================================================================
//   class Operand_
//=======================================================================


class ItemGroupParser::Operand_ : public Object
{
 public:

  inline                  Operand_ ();

  explicit inline         Operand_

    ( const IdxVector&      ivec );


 public:

  const IdxVector         iitems;

};


//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------


inline ItemGroupParser::Operand_::Operand_ ()
{}


inline ItemGroupParser::Operand_::Operand_

  ( const IdxVector&  ivec ) :

    iitems ( ivec )

{}


//=======================================================================
//   class ItemGroupParser::Data_
//=======================================================================


class ItemGroupParser::Data_ : public Collectable
{
 public:

  inline                    Data_        ();

  inline void               reset        ();
  inline void               clear        ();

  inline void               setGroup

    ( const Ref<XItemGroup>&  group );

  inline void               pushOperand  ();

  inline void               pushOperand

    ( const IdxVector&        iitems );



 public:

  Ref<ParserScope>          scope;
  Ref<ItemSet>              items;
  Ref<XItemGroup>           group;
  Ref<Dictionary>           groups;

  Flex<int>                 savedModes;
  ObjFlex                   operands;
  Flex<Operator_*>          operators;

  IdxVector                 ibuf;
  BoolVector                mask;

  int                       mode;
  bool                      mustFilter;
  bool                      skipUnknown;

};


//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


ItemGroupParser::Data_::Data_ ()
{
  mode        = 0;
  mustFilter  = false;
  skipUnknown = false;
}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


inline void ItemGroupParser::Data_::reset ()
{
  if ( items )
  {
    const idx_t  itemCount = items->size ();

    if ( ibuf.size() != itemCount )
    {
      ibuf.resize ( itemCount );
      ibuf = -1;
    }

    if ( mask.size() != itemCount )
    {
      mask.resize ( itemCount );
      mask = false;
    }
  }

  savedModes.clear ();
  operands  .clear ();
  operators .clear ();

  mustFilter = false;
  mode       = 0;
}


//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------


void ItemGroupParser::Data_::clear ()
{
  ibuf       = -1;
  mask       =  false;
  mustFilter =  false;
  mode       =  0;

  savedModes.clear ();
  operands  .clear ();
  operators .clear ();
}


//-----------------------------------------------------------------------
//   setGroup
//-----------------------------------------------------------------------


inline void ItemGroupParser::Data_::setGroup

  ( const Ref<XItemGroup>& group )

{
  if ( group )
  {
    items = group->getItems ();
  }
  else
  {
    items = nullptr;
  }

  this->group = group;
}


//-----------------------------------------------------------------------
//   pushOperand
//-----------------------------------------------------------------------


inline void ItemGroupParser::Data_::pushOperand ()
{
  operands.pushBack ( newInstance<Operand_>() );
}


inline void ItemGroupParser::Data_::pushOperand

  ( const IdxVector&  iitems )

{
  operands.pushBack ( newInstance<Operand_>( iitems ) );
}


//=======================================================================
//   class ItemGroupParser::Utils_
//=======================================================================


class ItemGroupParser::Utils_ : public ParseUtils
{
 public:

  enum                    Mode
  {
                            NULL_MODE,
                            OPERAND_MODE,
                            OPERATOR_MODE
  };


  static Operator_*       OR_OP;
  static Operator_*       AND_OP;
  static Operator_*       AND_NOT_OP;
  static Operator_*       NOT_OP;


  static void             initOnce ();

  static void             parse

    ( State&                state,
      Data_&                data );

  static void             parseList

    ( State&                state,
      Data_&                data );

  static void             scanList

    ( State&                state,
      Data_&                data );

  static void             pushGroup

    ( State&                state,
      Data_&                data,
      const String&         name );

  static inline void      toOperandMode

    ( State&                state,
      Data_&                data );

  static inline void      beginExpression

    ( Data_&                data );

  static void             endExpression

    ( State&                state,
      Data_&                data );

  static inline void      beginSubExpression

    ( State&                state,
      Data_&                data );

  static inline void      endSubExpression

    ( State&                state,
      Data_&                data );

  static void             pushOperator

    ( State&                state,
      Data_&                data,
      Operator_*            op   );

  static void             popOperator

    ( State&                state,
      Data_&                data );

  static Ref<Operand_>    popOperand

    ( State&                state,
      Data_&                data,
      const char*           op   );

  static void             updateGroup

    ( Data_&                data );

  static void             filterItems

    ( IdxVector&            iitems,
      const Scope&          scope,
      const ItemSet&        items );

  static inline void      logProgress

    ( State&                state,
      idx_t&                counter,
      const String&         itemName,
      idx_t                 groupSize );

  static void             noSuchItemError

    ( State&                state,
      idx_t                   itemID,
      const ItemSet&        items );

};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


ItemGroupParser::Operator_*
  ItemGroupParser::Utils_::OR_OP      = nullptr;
ItemGroupParser::Operator_*
  ItemGroupParser::Utils_::AND_OP     = nullptr;
ItemGroupParser::Operator_*
  ItemGroupParser::Utils_::AND_NOT_OP = nullptr;
ItemGroupParser::Operator_*
  ItemGroupParser::Utils_::NOT_OP     = nullptr;


//-----------------------------------------------------------------------
//   initOnce
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::initOnce ()
{
  OR_OP      =
    new Operator_ ( & Operator_::orFunc,     "|", 1,
                      Operator_::LEFT_TO_RIGHT );

  AND_OP     =
    new Operator_ ( & Operator_::andFunc,    "&", 2,
                      Operator_::LEFT_TO_RIGHT );

  AND_NOT_OP =
    new Operator_ ( & Operator_::andNotFunc, "\\",
                               3,
                               Operator_::LEFT_TO_RIGHT );

  NOT_OP     =
    new Operator_ ( & Operator_::notFunc,    "!", 4,
                      Operator_::RIGHT_TO_LEFT );
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::parse

  ( State&  state,
    Data_&  data )

{
  const Scope*  scope =   data.scope.get ();
  Tokenizer&    input = * state.input;

  int           token;


  data.reset ();

  beginExpression ( data );

  while ( true )
  {
    token = input.nextToken ();

    if ( token == '{' )
    {
      if ( data.group )
      {
        parseList ( state, data );
      }
      else
      {
        scanList  ( state, data );
      }
    }
    else if ( token == Tokenizer::SYMBOL_TOKEN )
    {
      pushGroup ( state, data, getSymbol( input, scope ) );
    }
    else if ( token == Tokenizer::STRING_TOKEN )
    {
      pushGroup ( state, data, getString( input, scope ) );
    }
    else if ( token == '(' )
    {
      beginSubExpression ( state, data );
    }
    else if ( token == ')' )
    {
      endSubExpression ( state, data );
    }
    else if ( token == '|' )
    {
      pushOperator ( state, data, OR_OP );
    }
    else if ( token == '&' )
    {
      pushOperator ( state, data, AND_OP );
    }
    else if ( token == '\\' )
    {
      pushOperator ( state, data, AND_NOT_OP );
    }
    else if ( token == '!' )
    {
      pushOperator ( state, data, NOT_OP );

      data.mustFilter = true;
    }
    else
    {
      break;
    }
  }

  endExpression ( state, data );

  if ( data.operands.size() > 1 )
  {
    parseError ( state, "missing `)\'" );
  }

  input.pushBackToken ();

  if ( data.group && data.operands.size() > 0 )
  {
    updateGroup ( data );

    data.operands.clear ();
  }
}


//-----------------------------------------------------------------------
//   parseList
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::parseList

  ( State&  state,
    Data_&  data )

{
  const ItemSet&  items    = * data .items;
  const String    itemName =   items.getItemName ();
  const ItemMap&  itemMap  = * items.getItemMap  ();
  const Scope*    scope    =   data .scope.get   ();
  Tokenizer&      input    = * state.input;

  BoolVector      mask     = data.mask;
  IdxVector       ibuf     = data.ibuf;

  idx_t           first, last, stride;

  idx_t           itemID;
  idx_t           iitem;
  int             token;
  idx_t           counter;
  idx_t           n;


  toOperandMode ( state, data );

  token   = 0;
  counter = n = 0_idx;

  while ( true )
  {
    counter++;

    token = input.nextToken ();

    if ( token == ',' )
    {
      continue;
    }
    else if ( token == Tokenizer::INTEGER_TOKEN )
    {
      itemID = getItemID  ( input, scope );
      iitem  = itemMap.findItem ( itemID );

      if ( iitem < 0 )
      {
        if ( ! data.skipUnknown )
        {
          noSuchItemError ( state, itemID, items );
        }
      }
      else if ( ! mask[iitem] )
      {
        ibuf[n++]   = iitem;
        mask[iitem] = true;
      }
    }
    else if ( token == '[' )
    {
      parseRange ( state, first, last, stride );

      for ( ; first < last; first += stride )
      {
        if ( scope )
        {
          itemID = scope->expandID ( first );
        }
        else
        {
          itemID = first;
        }

        iitem = itemMap.findItem ( itemID );

        if ( iitem < 0 )
        {
          if ( ! data.skipUnknown )
          {
            noSuchItemError ( state, itemID, items );
          }
        }
        else if ( ! mask[iitem] )
        {
          ibuf[n++]   = iitem;
          mask[iitem] = true;
        }
      }
    }
    else
    {
      break;
    }

    logProgress ( state, counter, itemName, n );
  }

  if ( token != '}' )
  {
    parseError ( state, "expected a `}\'" );
  }

  IdxVector result ( ibuf[slice(BEGIN,n)].clone() );

  mask[result] = false;

  data.pushOperand ( result );
}


//-----------------------------------------------------------------------
//   scanList
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::scanList

  ( State&  state,
    Data_&  data )

{
  const String  itemName =   "item";
  Tokenizer&    input    = * state.input;

  idx_t         first, last, stride;

  idx_t         groupSize;
  int           token;
  idx_t         counter;
  idx_t         n;


  toOperandMode ( state, data );

  counter = groupSize = 0;

  while ( true )
  {
    counter++;

    token = input.nextToken ();

    if ( token == ',' )
    {
      continue;
    }
    else if ( token == Tokenizer::INTEGER_TOKEN )
    {
      groupSize++;
    }
    else if ( token == '[' )
    {
      parseRange ( state, first, last, stride );

      n = last - first;

      if ( n > 0 )
      {
        n = 1 + (n - 1) / stride;
      }

      groupSize += n;
    }
    else
    {
      break;
    }

    logProgress ( state, counter, itemName, groupSize );
  }

  if ( token != '}' )
  {
    parseError ( state, "expected a `}\'" );
  }

  data.pushOperand ();
}


//-----------------------------------------------------------------------
//   pushGroup
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::pushGroup

  ( State&         state,
    Data_&         data,
    const String&  name )

{
  using jem::dynamicCast;

  Ref<ItemGroup>  group;


  toOperandMode ( state, data );

  if ( ! data.items )
  {
    data.pushOperand ();

    return;
  }

  if ( data.groups )
  {
    group = dynamicCast<ItemGroup> ( data.groups->get( name ) );

    if ( group && group->getItems() != data.items.get() )
    {
      group = nullptr;
    }
  }

  if ( ! group )
  {
    const String itemName = data.items->getItemName ();

    inputError (
      state,
      String::format (
        "undefined %s group: `%s\'", itemName, name
      )
    );
  }

  data.pushOperand ( group->getIndices() );
}


//-----------------------------------------------------------------------
//   toOperandMode
//-----------------------------------------------------------------------


inline void ItemGroupParser::Utils_::toOperandMode

  ( State&  state,
    Data_&  data )

{
  if ( data.mode == OPERAND_MODE )
  {
    parseError ( state, "expected an operator" );
  }

  data.mode = OPERAND_MODE;
}


//-----------------------------------------------------------------------
//   beginExpression
//-----------------------------------------------------------------------


inline void ItemGroupParser::Utils_::beginExpression

  ( Data_&  data )

{
  data.savedModes.pushBack ( data.mode );
  data.operands  .pushBack ( nullptr );
  data.operators .pushBack ( nullptr );

  data.mode = NULL_MODE;
}


//-----------------------------------------------------------------------
//   endExpression
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::endExpression

  ( State&  state,
    Data_&  data )

{
  JEM_PRECHECK2 ( data.savedModes.size() > 0 &&
                  data.operands  .size() > 0 &&
                  data.operators .size() > 0,
                  "invalid parser state" );


  if ( data.mode == OPERATOR_MODE )
  {
    parseError ( state, "invalid expression: missing operand" );
  }

  while ( data.operators.back() )
  {
    popOperator ( state, data );
  }

  data.operators.popBack ();

  Ref<Operand_>  result =

    static_cast<Operand_*> ( data.operands.back() );

  data.operands.popBack ();

  if ( result )
  {
    if ( data.operands.back() )
    {
      parseError ( state, "invalid expression: too many operands" );
    }

    data.operands.set ( data.operands.size() - 1, result );
  }
  else
  {
    data.pushOperand ();
  }

  data.mode = data.savedModes.back ();
  data.savedModes.popBack ();
}


//-----------------------------------------------------------------------
//   beginSubExpression
//-----------------------------------------------------------------------


inline void ItemGroupParser::Utils_::beginSubExpression

  ( State&  state,
    Data_&  data )

{
  toOperandMode   ( state, data );
  beginExpression ( data );
}


//-----------------------------------------------------------------------
//   endSubExpression
//-----------------------------------------------------------------------


inline void ItemGroupParser::Utils_::endSubExpression

  ( State&  state,
    Data_&  data )

{
  if ( data.savedModes.size() > 1 )
  {
    endExpression ( state, data );
  }
  else
  {
    parseError ( state, "unbalanced `)\'" );
  }
}


//-----------------------------------------------------------------------
//   pushOperator
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::pushOperator

  ( State&      state,
    Data_&      data,
    Operator_*  op )

{
  JEM_PRECHECK2 ( data.operators.size() > 0,
                  "empty operator stack" );

  Operator_*  last = data.operators.back ();


  if ( op->isUnary() && data.mode == OPERAND_MODE )
  {
    parseError ( state );
  }

  if ( op->isBinary() && data.mode != OPERAND_MODE )
  {
    parseError ( state );
  }

  while ( last && last->rank > op->rank )
  {
    popOperator ( state, data );
    last = data.operators.back ();
  }

  if ( op->assoc == Operator_::LEFT_TO_RIGHT )
  {
    while ( last && last->rank == op->rank )
    {
      popOperator ( state, data );
      last = data.operators.back ();
    }
  }

  data.operators.pushBack ( op );

  data.mode = OPERATOR_MODE;
}


//-----------------------------------------------------------------------
//   popOperator
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::popOperator

  ( State&  state,
    Data_&  data )

{
  JEM_PRECHECK2 ( data.operands .size() > 0 &&
                  data.operators.size() > 0,
                  "empty operator/operand stack" );

  Operator_*  op = data.operators.back ();


  data.operators.popBack ();

  if ( op->isUnary() )
  {
    Ref<Operand_>  rhs = popOperand ( state, data, op->name );

    data.pushOperand (
      op->func1 ( data.mask, data.ibuf, rhs->iitems )
    );
  }
  else
  {
    Ref<Operand_>  rhs = popOperand ( state, data, op->name );
    Ref<Operand_>  lhs = popOperand ( state, data, op->name );

    data.pushOperand (
      op->func2 ( data.mask, data.ibuf, lhs->iitems, rhs->iitems )
    );
  }
}


//-----------------------------------------------------------------------
//   popOperand
//-----------------------------------------------------------------------


Ref<ItemGroupParser::Operand_>

  ItemGroupParser::Utils_::popOperand

  ( State&       state,
    Data_&       data,
    const char*  op )

{
  JEM_PRECHECK2 ( data.operands.size() > 0, "empty operand stack" );

  Ref<Operand_>  operand =

    static_cast<Operand_*> ( data.operands.back() );

  if ( ! operand )
  {
    parseError (
      state,
      String::format ( "missing operand for operator `%s\'", op )
    );
  }

  data.operands.popBack ();

  return operand;
}


//-----------------------------------------------------------------------
//   updateGroup
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::updateGroup ( Data_& data )
{
  JEM_PRECHECK2 ( data.group, "NULL ItemGroup" );

  Ref<Operand_>  op     =

    static_cast<Operand_*> ( data.operands.back() );

  XItemGroup&    group  = * data.group;

  BoolVector     mask   = data.mask;
  IdxVector      ibuf   = data.ibuf;

  IdxVector      iitems = group.getIndices ();
  IdxVector      jitems = op->iitems;

  idx_t          jitem;
  idx_t          j, k, n;


  if ( data.scope && data.mustFilter )
  {
    filterItems ( jitems, *data.scope, *data.items );
  }

  k = iitems.size ();
  n = jitems.size ();

  if      ( k == 0 )
  {
    group.append ( jitems );
  }
  else if ( n > k / 8 )
  {
    mask[iitems] = true;

    for ( j = k = 0; j < n; j++ )
    {
      jitem = jitems[j];

      if ( ! mask[jitem] )
      {
        ibuf[k++] = jitem;
      }
    }

    mask[iitems] = false;

    group.append ( ibuf[slice(BEGIN,k)] );
  }
  else
  {
    group.filter ( jitems );

    for ( j = k = 0; j < n; j++ )
    {
      jitem = jitems[j];

      if ( jitem >= 0 )
      {
        ibuf[k++] = jitem;
      }
    }

    group.append ( ibuf[slice(BEGIN,k)] );
  }
}


//-----------------------------------------------------------------------
//   filterItems
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::filterItems

  ( IdxVector&      iitems,
    const Scope&    scope,
    const ItemSet&  items )

{
  const idx_t  n = iitems.size ();

  IdxVector    itemIDs ( n );

  idx_t        itemID;
  idx_t        i, j;


  items.getItemMap()->getItemIDs ( itemIDs, iitems );

  j = 0;

  for ( i = 0; i < n; i++ )
  {
    itemID = itemIDs[i];

    if ( scope.restrictID( itemID ) )
    {
      iitems[j++] = iitems[i];
    }
  }

  if ( j < n )
  {
    iitems.ref ( iitems[slice(BEGIN,j)] );
  }
}


//-----------------------------------------------------------------------
//   logProgress
//-----------------------------------------------------------------------


inline void ItemGroupParser::Utils_::logProgress

  ( State&         state,
    idx_t&         counter,
    const String&  itemName,
    idx_t          groupSize )

{
  using jem::xml::ParseLog;

  if ( counter > 1000_idx )
  {
    counter = 0;

    state.log->logEvent (
      state,
      ParseLog::PROGRESS,
      String::format (
        "%s group size: %d", itemName, groupSize
      )
    );
  }
}


//-----------------------------------------------------------------------
//   noSuchItemError
//-----------------------------------------------------------------------


void ItemGroupParser::Utils_::noSuchItemError

  ( State&          state,
    idx_t             itemID,
    const ItemSet&  items )

{
  String  itemName = items.getItemName ();

  inputError (
    state,
    String::format ( "undefined %s ID: %d", itemName, itemID )
  );
}


//=======================================================================
//   class ItemGroupParser
//=======================================================================

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


ItemGroupParser::ItemGroupParser

  ( const Ref<XItemGroup>&  group,
    const Ref<Dictionary>&  groups )

{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, & Utils_::initOnce );

  data_ = newInstance<Data_> ();

  data_->setGroup ( group );

  data_->groups = groups;
}


ItemGroupParser::~ItemGroupParser ()
{}


//-----------------------------------------------------------------------
//   reset
//-----------------------------------------------------------------------


void ItemGroupParser::reset ()
{
  data_->clear ();
}


//-----------------------------------------------------------------------
//   parse
//-----------------------------------------------------------------------


void ItemGroupParser::parse ( State& state )
{
  Data_&  d = * data_;

  try
  {
    Utils_::parse ( state, d );
  }
  catch ( const ItemIDException& ex )
  {
    d.clear ();

    if ( ! d.items )
    {
      inputError ( state, ex.what() );
    }
    else
    {
      ParseUtils::itemIDError ( state, *d.items, ex );
    }
  }
  catch ( ... )
  {
    d.clear ();
    throw;
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


void ItemGroupParser::takeAction

  ( const String&      action,
    const Properties&  params )

{
  Data_&  d = * data_;


  if ( action == ParserActions::NEW_SCOPE )
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
//   setGroup
//-----------------------------------------------------------------------


void ItemGroupParser::setGroup ( const Ref<XItemGroup>& group )

{
  data_->setGroup ( group );
}


//-----------------------------------------------------------------------
//   setGroups
//-----------------------------------------------------------------------


void ItemGroupParser::setGroups ( const Ref<Dictionary>& groups )
{
  data_->groups = groups;
}


JIVE_END_PACKAGE( util )
