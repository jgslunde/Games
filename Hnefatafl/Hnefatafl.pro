TEMPLATE = app
TARGET = Hnefatafl
INCLUDEPATH += .

# Input
SOURCES += main.cpp \
           HnefataflGui.cpp \
           HnefataflBoard.cpp
           HnefataflGame.cpp

HEADERS += HnefataflGui.h \
           HnefataflBoard.h
           HnefataflGame.h

QT += widgets