// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/io/Utils.h>
#include <nls/geometry/MetaShapeCamera.h>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

/**
 * Reads camera calibration information from our own camera json format and converts it to MetaShapeCameras.
 */
template <class T>
bool ReadMetaShapeCamerasFromJsonFile(const std::string& filename, std::vector<MetaShapeCamera<T>>& cameras);

/**
 * Writes camera calibration information into our own camera json format.
 */
template <class T>
bool WriteMetaShapeCamerasToJsonFile(const std::string& filename, const std::vector<MetaShapeCamera<T>>& cameras);

/**
 * Writes camera calibration information into xmp format.
 */
template<class T>
bool WriteMetaShapeCamerasToXmpFolder(const std::string& folderPath, const std::vector<MetaShapeCamera<T>>& cameras, int type);

/**
 * Reality Capture camera parameters as stored in XMP sidecar files.
 */
template<class T>
struct RealityCaptureCamera {
    Eigen::Matrix3<T> rotation;
    Eigen::Vector3<T> position;
    Eigen::VectorX<T> distortion;
    T focalLength35mm = T(0.0);
    T skew = T(0.0);
    T aspectRatio = T(0.0);
    T principalPointU = T(0.0);
    T principalPointV = T(0.0);
};

/**
 * Parses a Reality Capture XMP sidecar file into a RealityCaptureCamera.
 */
template <class T>
RealityCaptureCamera<T> ParseRealityCaptureXmp(const std::string& path);

/**
 * Converts a RealityCaptureCamera to a MetaShapeCamera given the image dimensions.
 */
template<typename T>
MetaShapeCamera<T> RealityCaptureToMetashapeCamera(T width, T height, const RealityCaptureCamera<T>& rc);

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
