add_library ( orthruscompute_amd64 OBJECT ${SOURCE_COMPUTE} )
target_compile_definitions ( orthruscompute_amd64 PUBLIC TARGET_PLATFORM_AMD64 )

set ( __PLATFORM amd64 )
include ( ../_targets.cmake )
