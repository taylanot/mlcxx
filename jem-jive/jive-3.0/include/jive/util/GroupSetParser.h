
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

#ifndef JIVE_UTIL_GROUPSETPARSER_H
#define JIVE_UTIL_GROUPSETPARSER_H

#include <jem/xml/Parser.h>
#include <jive/defines.h>


JIVE_BEGIN_PACKAGE( util )


class XGroupSet;
class ParserScope;


//-----------------------------------------------------------------------
//   class GroupSetParser
//-----------------------------------------------------------------------


class GroupSetParser : public jem::xml::Parser
{
 public:

  JEM_DECLARE_CLASS       ( GroupSetParser, jem::xml::Parser );


  explicit                  GroupSetParser

    ( const Ref<XGroupSet>&   groups    =  nullptr,
      idx_t                   groupSize = -1 );

  virtual void              parse

    ( State&                  state  )         override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )         override;

  void                      setGroups

    ( const Ref<XGroupSet>&   groups );

  void                      setGroupSize

    ( idx_t                   size   );


 protected:

  virtual                  ~GroupSetParser  ();


 private:

  class                     Utils_;

  Ref<ParserScope>          scope_;
  Ref<XGroupSet>            groups_;

  idx_t                     groupSize_;
  bool                      skipUnknown_;

};


JIVE_END_PACKAGE( util )

#endif
