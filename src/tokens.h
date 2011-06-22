#ifndef TOKENS_H
#define TOKENS_H

#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <iostream>

#define CR '\010'
#define LF '\0'

#define MAX_STRING_LENGTH 128

enum TokenIDs { 
  T_DOUBLE,
  T_COMMA,

  T_LEFT_ANGLE,
  T_RIGHT_ANGLE,
  T_LEFT_CURLY,
  T_RIGHT_CURLY,

  T_ROTATE,
  T_TRANSLATE,
  T_SCALE,
  T_MATRIX,

  T_POLYGON,
  T_SPHERE,
  T_BOX,
  T_CYLINDER,
  T_CONE,
  T_QUADRIC,
  T_TRIANGLE, // added by mbecker
  T_PLANE, // added by mbecker

  T_CAMERA,
  T_LOCATION,
  T_RIGHT,
  T_UP,
  
  T_LOOK_AT,
  T_ANGLE,

  T_GLOBAL_SETTINGS,
  T_AMBIENT_LIGHT,
  T_LIGHT_SOURCE,
  T_FINISH,
  T_PIGMENT,
  T_COLOR,
  T_RGB,
  T_RGBF,
  T_REFLECTION,
  T_REFRACTION, // added by mbecker
  T_AMBIENT,
  T_DIFFUSE,
  T_SPECULAR, // added by mbecker
  T_ROUGHNESS, // added by mbecker
  T_PHONG,
  T_METALLIC,
  T_PHONG_SIZE,
  T_INTERIOR,
  T_IOR,

  T_NULL,
  T_EOF,

  T_LAST
};

struct Token{ 
  enum TokenIDs id; 
  double double_value; /* has meaning only if id = T_DOUBLE */
};

struct ReservedWord { 
	enum TokenIDs id; 
	std::string str;
};

class Tokenizer{
	private:
	std::fstream & infile;
	bool unget_flag;
	int lineNumber;
	
	static std::string TokenNames[];
	static ReservedWord ReservedWords[];
	Token token;
	
	public:
	Tokenizer(std::fstream & file):
	infile(file),unget_flag(false),lineNumber(1)
	{
		token.id = T_NULL;
	};
	void Error(std::string error_msg);
	
	private:
	void SkipSpaces();
	void ReadDouble();
	enum TokenIDs FindReserved(const char* str);
	void ReadName();
	
	public:
	Token GetToken();
	void UngetToken();
};

#endif
