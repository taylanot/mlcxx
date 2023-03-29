
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

#ifndef JIVE_UTIL_POINTSETPARSER_H
#define JIVE_UTIL_POINTSETPARSER_H

#include <jem/xml/Parser.h>
#include <jem/defines.h>


JIVE_BEGIN_PACKAGE( util )


class XPointSet;
class ParserScope;


//-----------------------------------------------------------------------
//   class PointSetParser
//-----------------------------------------------------------------------


class PointSetParser : public jem::xml::Parser
{
 public:

  JEM_DECLARE_CLASS       ( PointSetParser, jem::xml::Parser );


  explicit                  PointSetParser

    ( const Ref<XPointSet>&   points =  nullptr,
      idx_t                   rank   = -1 );

  virtual void              parse

    ( State&                  state  )         override;

  virtual void              takeAction

    ( const String&           action,
      const Properties&       params )         override;

  void                      setPointSet

    ( const Ref<XPointSet>&   points );

  void                      setRank

    ( idx_t                   rank   );


 protected:

  virtual                  ~PointSetParser  ();


 private:

  void                      parse_

    ( State&                  state );

  void                      scan_

    ( State&                  state );


 private:

  class                     Utils_;

  Ref<ParserScope>          scope_;
  Ref<XPointSet>            points_;

  idx_t                     rank_;

};


JIVE_END_PACKAGE( util )

#endif
