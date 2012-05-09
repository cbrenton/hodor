#include "tokens.h"

using namespace std;

ReservedWord Tokenizer::ReservedWords[] = {
   {  T_ROTATE,    "rotate" },
   {  T_TRANSLATE, "translate" },
   {  T_SCALE,     "scale" },
   {  T_MATRIX,    "matrix" },
   {  T_POLYGON,   "polygon" },
   {  T_SPHERE,    "sphere" },
   {  T_BOX,       "box" },
   {  T_CYLINDER,  "cylinder" },
   {  T_CONE,      "cone" },
   {  T_QUADRIC,   "quadric" },
   {  T_TRIANGLE,  "triangle" }, // added by mbecker
   {  T_PLANE,     "plane" }, // added by mbecker
   {  T_CAMERA,    "camera" },
   {  T_LOCATION,  "location" },
   {  T_RIGHT,     "right" },
   {  T_UP,        "up" },
   {  T_LOOK_AT,   "look_at" },
   {  T_ANGLE,     "angle" },

   {  T_GLOBAL_SETTINGS, "global_settings"},
   {  T_AMBIENT_LIGHT, "ambient_light"},
   {  T_LIGHT_SOURCE, "light_source"},
   {  T_FINISH, "finish"},
   {  T_PIGMENT, "pigment"},
   {  T_RGB, "rgb"},
   {  T_COLOR, "color"},
   {  T_RGBF, "rgbf"},
   {  T_REFLECTION, "reflection"},
   {  T_REFRACTION, "refraction"}, // added by mbecker
   {  T_AMBIENT, "ambient"},
   {  T_DIFFUSE, "diffuse"},
   {  T_SPECULAR, "specular"}, // added by mbecker
   {  T_ROUGHNESS, "roughness"}, // added by mbecker
   {  T_PHONG, "phong"},
   {  T_METALLIC, "metallic"},
   {  T_PHONG_SIZE, "phong_size"},
   {  T_INTERIOR, "interior"},
   {  T_IOR, "ior"},
   {  T_LAST, ""}
};

string Tokenizer::TokenNames[] = {
   "DOUBLE",
   "COMMA",
   "LEFT_ANGLE",
   "RIGHT_ANGLE",
   "LEFT_CURLY",
   "RIGHT_CURLY",
   "ROTATE",
   "TRANSLATE",
   "SCALE",
   "MATRIX",
   "POLYGON",
   "SPHERE",
   "BOX",
   "CYLINDER",
   "CONE",
   "QUADRIC",
   "TRIANGLE", // added by mbecker
   "PLANE", // added by mbecker
   "CAMERA",
   "LOCATION",
   "RIGHT",
   "UP",
   "LOOK_AT",
   "ANGLE",
   "GLOBAL_SETTINGS",
   "AMBIENT_LIGHT",
   "LIGHT_SOURCE",
   "FINISH",
   "PIGMENT",
   "COLOR",
   "RGB",
   "RGBF",
   "REFLECTION",
   "REFRACTION", // added by mbecker
   "AMBIENT",
   "DIFFUSE",
   "SPECULAR", // added by mbecker
   "ROUGHNESS", // added by mbecker
   "PHONG",
   "METALLIC",
   "PHONG_SIZE",
   "INTERIOR",
   "IOR",
   "NULL",
   "EOF"
};

void Tokenizer::Error(string str){
   // printf("Line %d: %s\n", lineNumber, str.c_str());
   cout << "Line " << lineNumber << ": " << str << endl;
   exit(0);
}

void Tokenizer::SkipSpaces() {
   int c;
   while(1) {
      // c = getc(token.infile);
      c = infile.get();;
      if( c == '\n') lineNumber++;
      if (c == EOF ) return;
      if( c ==  '/'){
         /* we use slash only as a part of the
            comment begin seqence; if something other than another
            slash follows it, it is an error */
         if( infile.get() == '/'){
            /* skip everything till the end of the line */
            while( c != '\n' && c != '\r' && c != EOF ){
               // c = getc(token.infile);
               c = infile.get();
            }
            lineNumber++;
         } else Error("Missing secnd slash in comment");
      }
      if( !isspace(c))
         break;
   }
   // ungetc(c,token.infile);
   infile.putback((char)c);
}

void Tokenizer::ReadDouble() {
   /* this is cheating -- we'd better parse the number definition
      ourselves, to make sure it conforms to a known standard and
      to do error hanndling properly,
      but for our purposes  this is good enough */
   // int res;
   // res = fscanf( token.infile, "%le", &token.double_value);
   // if( res == 1 ) token.id = T_DOUBLE;
   infile >> token.double_value;
   if(!infile.fail()) token.id = T_DOUBLE;
   else Error("Could not read a number");
}

#define MAX_STR_LENGTH 128

/* this is a very stupid way of looking up things in a table,
   but because our table is so small, it works ok */

enum TokenIDs Tokenizer::FindReserved(const char* str) {
   struct ReservedWord * ptr;
   ptr = ReservedWords;
   while( ptr->id != T_LAST ){
      if( !strcmp(str, (ptr->str).c_str()) ) return ptr->id;
      ptr++;
   }
   return T_NULL;
}

void Tokenizer::ReadName() {
   /*
      char str[MAX_STR_LENGTH];
      int str_index;
      int c;

      str_index = 0;
      while (1){
      c = getc(token.infile);
      if (c == EOF) Error("Could not read a name");
      if (isalpha(c) || isdigit(c) || c == '_'){
   // if the name is too long, ignore extra characters
   if( str_index < MAX_STR_LENGTH-1)
   str[str_index++] = c;
   } else{
   ungetc(c,token.infile);
   break;
   }
   }
   str[str_index++] = '\0';
   */

   string str;
   infile >> str;
   if(infile.eof()) Error("Could not read a name");

   token.id = FindReserved(str.c_str());
   if( token.id == T_NULL) {
      fprintf(stderr, "%s: ", str.c_str());
      Error("Unknown reserved word");
   }
   return;
}

/* sets the global struct Token to the
   next input token, if there is one.
   if there is no legal token in the input,
   returns 0, otherwise returns 1.
   */

Token Tokenizer::GetToken() {
   int c;
   if( unget_flag) {
      unget_flag = false;
      return token;
   }

   SkipSpaces();
   // c = getc(token.infile);
   c = infile.get();

   if( c == EOF ) {
      token.id = T_EOF;
      return token;
   }

   if(isalpha(c)) {
      // ungetc(c,token.infile);
      infile.putback((char)c);
      ReadName();
   } else if( isdigit(c) || c == '.' || c == '-' || c == '+' ) {
      // ungetc(c,token.infile);
      infile.putback((char)c);
      ReadDouble();
   } else {
      switch(c) {
      case ',':
         token.id = T_COMMA;
         break;
      case '{':
         token.id = T_LEFT_CURLY;
         break;
      case '}':
         token.id = T_RIGHT_CURLY;
         break;
      case '<':
         token.id = T_LEFT_ANGLE;
         break;
      case '>':
         token.id = T_RIGHT_ANGLE;
         break;
      default:
         Error("Unknown token");
      }
   }
   return token;
}

/* Assumes that GetToken() was called at least once.
   Cannot be called two times without a GetToken() between
   the calls */
void Tokenizer::UngetToken() {
   assert(!unget_flag);
   unget_flag = true;
}
