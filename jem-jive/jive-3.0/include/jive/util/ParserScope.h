
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

#ifndef JIVE_UTIL_PARSERSCOPE_H
#define JIVE_UTIL_PARSERSCOPE_H

#include <jem/base/Object.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


//-----------------------------------------------------------------------
//   ParserScope
//-----------------------------------------------------------------------


class ParserScope : public Object
{
 public:

  JEM_DECLARE_CLASS     ( ParserScope, Object );


  virtual idx_t           expandID

    ( idx_t                 id   )       const = 0;

  virtual bool            restrictID

    ( idx_t&                id   )       const = 0;

  virtual String          expandName

    ( const String&         name )       const = 0;

  virtual bool            restrictName

    ( String&               name )       const = 0;

  inline bool             isGlobal    () const;


 protected:

  virtual                ~ParserScope ();

};


//-----------------------------------------------------------------------
//   class GlobalScope
//-----------------------------------------------------------------------


class GlobalScope : public ParserScope
{
 public:

  JEM_DECLARE_CLASS     ( GlobalScope, ParserScope );


                          GlobalScope ();

  virtual idx_t           expandID

    ( idx_t                 id   )       const override;

  virtual bool            restrictID

    ( idx_t&                id   )       const override;

  virtual String          expandName

    ( const String&         name )       const override;

  virtual bool            restrictName

    ( String&               name )       const override;


 private:

  virtual                ~GlobalScope ();

};




//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class ParserScope
//=======================================================================


inline bool ParserScope::isGlobal () const
{
  return (dynamic_cast<const GlobalScope*>( this ) != 0);
}


JIVE_END_PACKAGE( util )

#endif
