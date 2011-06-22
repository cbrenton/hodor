# Original SDL-GL-basic Makefile template written by Hans de Ruiter
# Modified by Chris Brenton
#
# License:
# This source code can be used and/or modified without restrictions.
# It is provided as is and the author disclaims all warranties, expressed 
# or implied, including, without limitation, the warranties of
# merchantability and of fitness for any purpose. The user must assume the
# entire risk of using the Software.

CC     = g++
GPU_CC = nvcc
CP     = cp
RM     = rm -rf
KILL   = killall -9
SHELL  = /bin/sh
MAKE   = make

LIBFLAGS = -I ./lib/
DEBUG = -ggdb
OPTIMIZE = -O3
ERROR = -Wconversion -Werror
CFLAGS = $(OPTIMIZE) -Wall -c $(DEBUG) $(ERROR) $(LIBFLAGS)
LDFLAGS = $(OPTIMIZE) $(DEBUG) $(ERROR)

TARGET = terlR
INPUTEXT=pov
INPUTDIR=input
INPUTFILE=simple
OUTPUTDIR=output
ARGS = -w 320 -h 240 -i $(INPUTDIR)/$(INPUTFILE).$(INPUTEXT)

# Additional linker libraries
LIBS = $(LIBFLAGS)

# -------------------------------------------------------------
# Nothing should need changing below this line

# The source files
SRCS = $(wildcard src/*.cpp)
#SRCS = $(wildcard *.cpp)

OBJS = $(SRCS:.cpp=.o)

# Rules for building
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

run:
	./$(TARGET) $(ARGS)

gdb:
	gdb --args ./$(TARGET) $(ARGS)

valgrind:
	valgrind --tool=memcheck --leak-check=full ./$(TARGET) $(ARGS)

clean:
	$(RM) $(TARGET) $(OBJS)

killall:
	$(KILL) $(TARGET)
