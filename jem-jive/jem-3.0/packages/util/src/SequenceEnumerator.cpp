
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jem, a general purpose programming toolkit.
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
 *
 *  Jem version: 3.0
 *  Date:        Fri 20 Dec 14:27:58 CET 2019
 */


#include <jem/base/ClassTemplate.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/util/SequenceEnumerator.h>


JEM_DEFINE_CLASS( jem::util::SequenceEnumerator );


JEM_BEGIN_PACKAGE( util )


//=======================================================================
//   class SequenceEnumerator
//=======================================================================

//-----------------------------------------------------------------------
//   destructor
//-----------------------------------------------------------------------


SequenceEnumerator::~SequenceEnumerator ()
{}


//-----------------------------------------------------------------------
//   toNext
//-----------------------------------------------------------------------


void SequenceEnumerator::toNext ()
{
  move ( 1 );
}


//-----------------------------------------------------------------------
//   toPrevious
//-----------------------------------------------------------------------


void SequenceEnumerator::toPrevious ()
{
  move ( -1_idx );
}


//-----------------------------------------------------------------------
//   writeError_
//-----------------------------------------------------------------------


void SequenceEnumerator::writeError_ () const
{
  Class*  type = getClass ();

  throw IllegalOperationException (
    type->getName (),
    "an enumerator may not modify its sequence while another "
    "enumerator is active"
  );
}


//-----------------------------------------------------------------------
//   moveError_
//-----------------------------------------------------------------------


void SequenceEnumerator::moveError_ ( idx_t count ) const
{
  Class*  type = getClass ();
  String  what;

  if ( count >= 0 )
  {
    what = "enumerator moved past the end of its sequence";
  }
  else
  {
    what = "enumerator moved past the begin of its sequence";
  }

  throw IllegalOperationException ( type->getName(), what );
}


//-----------------------------------------------------------------------
//   atEndError_
//-----------------------------------------------------------------------


void SequenceEnumerator::atEndError_ () const
{
  Class*  type = getClass ();

  throw IllegalOperationException (
    type->getName (),
    "enumerator is at the end of its sequence"
  );
}


JEM_END_PACKAGE( util )
