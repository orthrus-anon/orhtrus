protobuf_generate_cpp(PROTO_SRC PROTO_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/orthrus.proto)
add_library(orthrus_proto ${PROTO_HEADER} ${PROTO_SRC})
target_compile_options(orthrus_proto PRIVATE -Wno-error)
