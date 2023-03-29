//#include <cctype>
//#include <jem/io/FileReader.h>
//#include <jem/io/File.h>
//#include <jem/base/System.h>
//
//using jem::System;
//using jem::String;
//
//int read ()
//{
//	using namespace jem;
//	using namespace jem::io;
//
//	Ref<Reader> input;
//	String str;
//	int i, c;
//
//	input = newInstance<FileReader>("input.msh");
//
//	for (int j = 0; j < 5; j++)
//	{
//
//		c = input -> read ();
//
//		if (std::isdigit(c))
//		{
//			input -> pushBack(c);
//			*input >> i;
//		}
//		else
//		{
//			*input >> str;
//		}
//		System::out() << str << io::endl;
//	}
//	return 0; 
//}

#include <jem/base/System.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/Array.h>
#include <jem/io/FileWriter.h>
#include <jem/io/File.h>
#include <jem/io/FileReader.h>


using namespace jem;
using namespace jem::io;


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


int run ()
{
  Ref<Writer>     out = newInstance<FileWriter> ( "output.data" );
  Ref<Reader>     in  = newInstance<FileReader> ( "input.msh"  );

  Array<double>   value ( 10 );


  for ( idx_t i = 0; i < value.size(); i++ )
  {
    readLine ( *in, value[i] );
  }

  sort  ( value );
  print ( *out, value );

  out->flush ();

  return 0;
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ()
{
  return System::exec ( & run );
}

//int main ()
//{
//  return System::exec ( read );
//}
