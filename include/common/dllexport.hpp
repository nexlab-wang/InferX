#pragma once

#ifdef _WIN32
#ifdef MODEL_INFER_EXPORT
#define MODEL_INFER_API __declspec(dllexport)
#else
#define MODEL_INFER_API __declspec(dllimport)
#endif
#else
#define MODEL_INFER_API __attribute__((visibility("default")))
#endif