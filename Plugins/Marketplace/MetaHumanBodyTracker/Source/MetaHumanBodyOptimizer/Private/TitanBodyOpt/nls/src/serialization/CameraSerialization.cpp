// Copyright Epic Games, Inc. All Rights Reserved.

#include <nls/serialization/CameraSerialization.h>

#include <carbon/io/JsonIO.h>
#include <fstream>
#include <regex>
#include <sstream>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

template <class T>
bool ReadMetaShapeCamerasFromJsonFile(const std::string& filename, std::vector<MetaShapeCamera<T>>& cameras)
{
    JsonElement json = ReadJson(ReadFile(filename));
    if (!json.IsArray()) {
        LOG_ERROR("json camera file should contains an array as top level");
        return false;
    }

    std::vector<MetaShapeCamera<T>> newCameras;
    int cameraCounter = 0;
    for (int i = 0; i < static_cast<int>(json.Size()); ++i) {
        if (!json[i].Contains("metadata")) {
            LOG_ERROR("json camera file should contain an array of objects that have a metadata field");
            return false;
        }

        if (json[i]["metadata"]["type"].Get<std::string>() == "camera") {
            // we have a camera type
            MetaShapeCamera<T> camera;
            camera.SetCameraID(cameraCounter);
            cameraCounter += 1;

            camera.SetLabel(json[i]["metadata"]["name"].Get<std::string>());
            camera.SetWidth(json[i]["image_size_x"].Get<int>());
            camera.SetHeight(json[i]["image_size_y"].Get<int>());
            Eigen::Matrix<T, 3, 3> intrinsics = Eigen::Matrix<T, 3, 3>::Identity();
            if (json[i].Contains("fx") && json[i].Contains("fy") && json[i].Contains("cx") && json[i].Contains("cy")) {
                intrinsics(0, 0) = json[i]["fy"].Get<T>();
                intrinsics(1, 1) = json[i]["fy"].Get<T>();
                intrinsics(0, 2) = json[i]["cx"].Get<T>();
                intrinsics(1, 2) = json[i]["cy"].Get<T>();
                camera.SetIntrinsics(intrinsics);
            } else {
                LOG_ERROR("camera calibration is missing one of fx, fy, cx, cy");
                return false;
            }

            Eigen::Matrix<T, 4, 4> extrinsics;
            for (int c = 0; c < 4; ++c) {
                for (int r = 0; r < 4; ++r) {
                    extrinsics(r, c) = json[i]["transform"][4 * r + c].Get<T>();
                }
            }

            camera.SetExtrinsics(extrinsics);

            if (json[i]["distortion_model"].Get<std::string>() == "opencv") {
                auto valueOrNull = [](const JsonElement& j, const char* label) { return j.Contains(label) ? j[label].Get<T>() : T(0); };
                const T k1 = valueOrNull(json[i], "k1");
                const T k2 = valueOrNull(json[i], "k2");
                const T k3 = valueOrNull(json[i], "k3");
                const T k4 = valueOrNull(json[i], "k4");
                const T k5 = valueOrNull(json[i], "k5");
                const T k6 = valueOrNull(json[i], "k6");
                const T p1 = valueOrNull(json[i], "p1");
                const T p2 = valueOrNull(json[i], "p2");
                const T b1 = (json[i]["fx"].Get<T>() - json[i]["fy"].Get<T>());
                const T b2 = valueOrNull(json[i], "s");
                if ( k5 != T(0) || k6 != T(0)) {
                    LOG_ERROR("metashape camera does not support k5, and k6 parameter");
                    return false;
                }
                if (k4 != T(0)) {
                    camera.SetRadialDistortion(Eigen::Vector<T, 4>(k1, k2, k3, k4));
                }
                else {
                    camera.SetRadialDistortion(Eigen::Vector<T, 4>(k1, k2, k3, T(0)));
                }
                camera.SetTangentialDistortion(Eigen::Vector<T, 4>(p2, p1, T(0), T(0))); // note that metashape camera has swapped tangential distortion compared to opencv
                camera.SetSkew(Eigen::Vector<T, 2>(T(b1), T(b2)));
            } else {
                LOG_ERROR("no valid distortion model defined");
                return false;
            }

            newCameras.push_back(camera);
        }
    }

    cameras.swap(newCameras);
    return true;
}

template bool ReadMetaShapeCamerasFromJsonFile(const std::string& filename, std::vector<MetaShapeCamera<float>>& cameras);
template bool ReadMetaShapeCamerasFromJsonFile(const std::string& filename, std::vector<MetaShapeCamera<double>>& cameras);

template <class T>
bool WriteMetaShapeCamerasToJsonFile(const std::string& filename, const std::vector<MetaShapeCamera<T>>& cameras)
{
    JsonElement json(JsonElement::JsonType::Array);
    for (const MetaShapeCamera<T>& camera : cameras) {

        JsonElement jsonMeta(JsonElement::JsonType::Object);
        jsonMeta.Insert("type", JsonElement("camera"));
        jsonMeta.Insert("version", JsonElement(0));
        jsonMeta.Insert("name", JsonElement(camera.Label()));
        jsonMeta.Insert("camera", JsonElement(camera.Label()));

        JsonElement jsonCamera(JsonElement::JsonType::Object);
        jsonCamera.Insert("metadata", std::move(jsonMeta));

        jsonCamera.Insert("image_size_x", JsonElement(camera.Width()));
        jsonCamera.Insert("image_size_y", JsonElement(camera.Height()));

        if (camera.Skew().squaredNorm() > 0) {
            jsonCamera.Insert("fx", JsonElement(camera.Intrinsics()(0, 0) + camera.Skew()[0]));
            jsonCamera.Insert("fy", JsonElement(camera.Intrinsics()(1, 1)));
            jsonCamera.Insert("s", JsonElement(camera.Skew()[1]));
        }
        else {
            jsonCamera.Insert("fx", JsonElement(camera.Intrinsics()(0, 0)));
            jsonCamera.Insert("fy", JsonElement(camera.Intrinsics()(1, 1)));
        }
        jsonCamera.Insert("cx", JsonElement(camera.Intrinsics()(0,2)));
        jsonCamera.Insert("cy", JsonElement(camera.Intrinsics()(1,2)));

        if (camera.Intrinsics()(0, 1) != T(0)) {
            LOG_ERROR("failed to write camera parameters as intrinsics skew is not supported");
            return false;
        }

        jsonCamera.Insert("distortion_model", JsonElement("opencv"));
        jsonCamera.Insert("k1", JsonElement(camera.RadialDistortion()[0]));
        jsonCamera.Insert("k2", JsonElement(camera.RadialDistortion()[1]));
        jsonCamera.Insert("k3", JsonElement(camera.RadialDistortion()[2]));

        if (camera.RadialDistortion()[3] != T(0)) {
            jsonCamera.Insert("k4", JsonElement(camera.RadialDistortion()[3]));
        }

        jsonCamera.Insert("p1", JsonElement(camera.TangentialDistortion()[1])); // note swapped tangential distortion between opencv and metashape
        jsonCamera.Insert("p2", JsonElement(camera.TangentialDistortion()[0]));
        if (camera.TangentialDistortion()[2] != T(0) || camera.TangentialDistortion()[3] != T(0)) {
            LOG_ERROR("failed to write camera parameters as extended tangential distortion is not supported");
            return false;
        }

        JsonElement jsonTransform(JsonElement::JsonType::Array);
        const Eigen::Matrix<T, 4, 4> transform = camera.Extrinsics().Matrix();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                jsonTransform.Append(JsonElement(transform(r, c)));
            }
        }
        jsonCamera.Insert("transform", std::move(jsonTransform));
        json.Append(std::move(jsonCamera));
    }

    WriteFile(filename, WriteJson(json, /*tabs=*/1));

    return true;
}

template bool WriteMetaShapeCamerasToJsonFile(const std::string& filename, const std::vector<MetaShapeCamera<float>>& cameras);
template bool WriteMetaShapeCamerasToJsonFile(const std::string& filename, const std::vector<MetaShapeCamera<double>>& cameras);

template<class T>
static void writeXmp(const std::string &filename,
              const std::string &calibrationPrior,
              const int group,
              T rcFocalLength,
              T principalPointU,
              T principalPointV,
              T skew,
              T aspectRatio,
              const Eigen::Vector<T, 4> &radialDistortion,
              const Eigen::Vector<T, 4> &tangentialDistortion,
              const Eigen::Matrix3<T> &rotation,
              const Eigen::Vector3<T> &translation)
{
    std::ofstream file;
    file.open(filename);

    auto position = -rotation.transpose() * translation;

    file << "<x:xmpmeta xmlns:x=\"adobe:ns:meta / \">" << std::endl;
    file << "  ";
    file << "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">" << std::endl;
    file << "    ";
    file << "<rdf:Description xcr:Version=\"3\" xcr:PosePrior=\"" << calibrationPrior << "\""
         << " xcr:DistortionPrior=\"" << calibrationPrior << "\""
         << " xcr:Coordinates=\"absolute\"" << std::endl;
    file << "       ";
    file << "xcr:DistortionModel=\"brown3t2\"" << std::endl;
    file << "       ";
    file << "xcr:FocalLength35mm=\"" << std::setprecision(8) << rcFocalLength << "\" xcr:Skew=\"" << skew << "\"" << std::endl;
    file << "       ";
    file << "xcr:AspectRatio=\"" << std::setprecision(8) << aspectRatio << "\" xcr:PrincipalPointU=\"" << std::setprecision(8) << principalPointU
         << "\"" << std::endl;
    file << "       ";
    file << "xcr:PrincipalPointV=\"" << std::setprecision(8) << principalPointV << "\" xcr:CalibrationPrior=\"" << calibrationPrior << "\""
         << std::endl;
    file << "       ";
    file << "xcr:CalibrationGroup=\"" << group << "\" xcr:DistortionGroup=\"" << group << "\""
         << " xcr:LockedPoseGroup=\"" << group << "\""
         << " xcr:InTexturing=\"" << group << "\"" << std::endl;
    file << "       ";
    file << "xcr:InMeshing=\"" << group << "\" xmlns:xcr=\"http://www.capturingreality.com/ns/xcr/1.1#\">" << std::endl;
    file << "      ";
    file << "<xcr:Rotation>";
    for (int j = 0; j < rotation.rows(); j++)
    {
        for (int k = 0; k < rotation.cols(); k++)
        {
            if ((j == 2) && (k == 2))
            {
                file << std::setprecision(8) << std::to_string(rotation(j, k));
            }
            else
            {
                file << std::setprecision(8) << rotation(j, k) << " ";
            }
        }
    }
    file << "</xcr:Rotation>" << std::endl;
    file << "      ";
    file << "<xcr:Position>";
    for (int j = 0; j < position.size(); j++)
    {
        if (j == 2)
        {
            file << std::setprecision(8) << std::to_string(position(j));
        }
        else
        {
            file << std::setprecision(8) << position(j) << " ";
        }
    }
    file << "</xcr:Position>" << std::endl;
    file << "      ";
    file << "<xcr:DistortionCoeficients>";
    file << radialDistortion(0) << " " << radialDistortion(1) << " " << radialDistortion(2) << " " << radialDistortion(3) << " "
         << tangentialDistortion(0) << " " << tangentialDistortion(1) << "</xcr:DistortionCoeficients>" << std::endl;
    file << "    ";
    file << "</rdf:Description>" << std::endl;
    file << "  ";
    file << "</rdf:RDF>" << std::endl;
    file << "</x:xmpmeta>" << std::endl;

    file.close();
}

template<class T>
bool WriteMetaShapeCamerasToXmpFolder(const std::string& folderPath, const std::vector<MetaShapeCamera<T>>& cameras, int type)
{
    std::string calibrationPrior = "exact";

    if (type == 0)
    {
        calibrationPrior = "initial";
    }

    const int group = 1;
    const T rcSensorWidth = 36.0;

    for (const auto &camera : cameras)
    {
        T f = camera.Intrinsics()(0, 0) + camera.Skew()(0);

        T cameraImageWidth = (T)(std::max(camera.Width(), camera.Height()));
        T rcPixelSize = rcSensorWidth / cameraImageWidth;
        T rcFocalLength = f * rcPixelSize;
        T aspectRatio = camera.Intrinsics()(1, 1) / f;

        T principalPointU = (camera.Intrinsics()(0, 2) - (T)camera.Width() / T(2)) / cameraImageWidth;
        T principalPointV = (camera.Intrinsics()(1, 2) - (T)camera.Height() / T(2)) / cameraImageWidth;
        const T skew = T(camera.Skew()(1) / cameraImageWidth);
        Eigen::Matrix3<T> rotation = camera.Extrinsics().Linear();
        Eigen::Vector3<T> translation = camera.Extrinsics().Translation();
        auto radialDistortion = camera.RadialDistortion();
        auto tangentialDistortion = camera.TangentialDistortion();
        std::string filename = folderPath + camera.Label() + ".xmp";
        writeXmp<T>(filename,
                    calibrationPrior,
                    group,
                    rcFocalLength,
                    principalPointU,
                    principalPointV,
                    skew,
                    aspectRatio,
                    radialDistortion,
                    tangentialDistortion,
                    rotation,
                    translation);
    }

    return true;
}

template bool WriteMetaShapeCamerasToXmpFolder(const std::string& folderPath, const std::vector<MetaShapeCamera<float>>& cameras, int type);
template bool WriteMetaShapeCamerasToXmpFolder(const std::string& folderPath, const std::vector<MetaShapeCamera<double>>& cameras, int type);

static bool ExtractAttr(const std::string& s, const std::string& key, std::string& out)
{
    std::string pattern = key + "=\"([^\"]*)\"";
    std::regex r(pattern, std::regex_constants::ECMAScript);
    std::smatch m;
    if (std::regex_search(s, m, r))
    {
        out = m[1];
        return true;
    }
    return false;
}

static bool ExtractTag(const std::string& s, const std::string& tag, std::string& out)
{
    std::string pattern = "<" + tag + R"(>\s*([^<]*)\s*</)" + tag + ">";
    std::regex r(pattern, std::regex_constants::ECMAScript);
    std::smatch m;
    if (std::regex_search(s, m, r))
    {
        out = m[1];
        return true;
    }
    return false;
}

static std::vector<double> ParseDoubles(const std::string& s)
{
    std::vector<double> v;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) v.push_back(std::stod(tok));
    return v;
}

static double ParseDouble(const std::string& s) { return s.empty() ? 0.0 : std::stod(s); }

template <class T>
RealityCaptureCamera<T> ParseRealityCaptureXmp(const std::string& path)
{
    std::string xml = ReadFile(path);
    RealityCaptureCamera<T> c;
    c.rotation.setIdentity();
    c.position.setZero();
    c.distortion = Eigen::VectorX<T>::Zero(6);
    std::string v;
    if (ExtractAttr(xml, "xcr:FocalLength35mm", v)) c.focalLength35mm = static_cast<T>(ParseDouble(v));
    if (ExtractAttr(xml, "xcr:Skew", v)) c.skew = static_cast<T>(ParseDouble(v));
    if (ExtractAttr(xml, "xcr:AspectRatio", v)) c.aspectRatio = static_cast<T>(ParseDouble(v));
    if (ExtractAttr(xml, "xcr:PrincipalPointU", v)) c.principalPointU = static_cast<T>(ParseDouble(v));
    if (ExtractAttr(xml, "xcr:PrincipalPointV", v)) c.principalPointV = static_cast<T>(ParseDouble(v));
    if (ExtractTag(xml, "xcr:Rotation", v))
    {
        auto d = ParseDoubles(v);
        if (d.size() >= 9)
        {
            Eigen::Matrix<T, 3, 3> Rr;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Rr(i, j) = (T)d[i * 3 + j];
                }
            }
            c.rotation = Rr;
        }
    }
    if (ExtractTag(xml, "xcr:Position", v))
    {
        auto d = ParseDoubles(v);
        if (d.size() >= 3)
        {
            c.position << (T)d[0], (T)d[1], (T)d[2];
        }
    }
    if (ExtractTag(xml, "xcr:DistortionCoeficients", v))
    {
        auto d = ParseDoubles(v);
        for (size_t i = 0; i < std::min(d.size(), static_cast<size_t>(c.distortion.size())); i++)
        {
            c.distortion(i) = (T)d[i];
        }
    }
    return c;
}

template RealityCaptureCamera<float> ParseRealityCaptureXmp(const std::string& path);
template RealityCaptureCamera<double> ParseRealityCaptureXmp(const std::string& path);

template<typename T>
MetaShapeCamera<T> RealityCaptureToMetashapeCamera(T width, T height, const RealityCaptureCamera<T>& rc)
{
    const T rcSensorWidth = T(36);
    T cameraImageWidth = std::max(width, height);
    T rcPixelSize = rcSensorWidth / cameraImageWidth;
    T f = rc.focalLength35mm / rcPixelSize;
    Eigen::Matrix3<T> K = Eigen::Matrix3<T>::Identity();
    K(0, 0) = f;
    K(1, 1) = rc.aspectRatio * f;
    K(0, 2) = rc.principalPointU * cameraImageWidth + width / T(2);
    K(1, 2) = rc.principalPointV * cameraImageWidth + height / T(2);

    Eigen::Vector2<T> s = Eigen::Vector2<T>::Zero();
    s[1] = rc.skew * cameraImageWidth;
    Eigen::Matrix3<T> R = rc.rotation;
    Eigen::Vector3<T> C = rc.position;
    Eigen::Vector3<T> t = -R * C;

    Affine<T, 3, 3> extrinsics;
    extrinsics.SetLinear(R);
    extrinsics.SetTranslation(t);
    MetaShapeCamera<T> out;
    out.SetIntrinsics(K);
    out.SetSkew(s);
    out.SetExtrinsics(extrinsics);

    Eigen::Vector4<T> radDist = Eigen::Vector4<T>::Zero();
    radDist(0) = rc.distortion(0);
    radDist(1) = rc.distortion(1);
    radDist(2) = rc.distortion(2);
    out.SetRadialDistortion(radDist);

    Eigen::Vector4<T> dist = Eigen::Vector4<T>::Zero();
    dist(0) = rc.distortion(3);
    dist(1) = rc.distortion(4);
    out.SetTangentialDistortion(dist);

    return out;
}

template MetaShapeCamera<float> RealityCaptureToMetashapeCamera(float width, float height, const RealityCaptureCamera<float>& rc);
template MetaShapeCamera<double> RealityCaptureToMetashapeCamera(double width, double height, const RealityCaptureCamera<double>& rc);

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
