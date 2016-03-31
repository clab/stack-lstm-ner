PROJECT(cnn:parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(lstm-parse lstm-parse.cc)
target_link_libraries(lstm-parse cnn ${Boost_LIBRARIES})

