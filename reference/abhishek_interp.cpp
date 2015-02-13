#include "stdafx.h"
#include "FlatImgFileRotaionFilter.h"
#include "FlatImgFileRotaionFilter.h"
namespace ImageServerReader
{
float Reflection(float v, int size)
{
    if (v < 0.0)
    {
        v = -floor(v);
        v = (float)((int)v % (2*size-2));
    }
    if (v >= size)
    {
        v = 2*size - 2 - v;
    }
    return v;
}
    
BOOL CFlatImgFileRotaionFilter::Init()
{
    CFlatImgFileFilterBase::Init();

    const float s_angles[ROTATION_VARIETY] = {-10.0f, -8.0f, -6.0f, -4.0f, -2.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    int imH = GetImageHeight();
    int imW = GetImageWidth();
    int cy =  imH / 2;
    int cx =  imW / 2;
    for (int i = 0; i < ROTATION_VARIETY; i++)
    {
        int *pCurrentMapping = _ppOffsetMapping[i] ;
        float pi = 3.1415926535f;
        float angle = s_angles[i] / 180.0f * pi;
        float cosAngle = cos(angle);
        float sinAngle = sin(angle);
        
        for (int y = 0; y < imH; y++)
        {
            for (int x = 0; x < imW; x++)
            {
                float u = (cosAngle * (float)(x - cx) + sinAngle * (float)(y - cy)) + cx;
                float v = ((0.0f - sinAngle) * (float)(x - cx) + cosAngle * (float)(y - cy)) + cy;
            
                // Reflecting u and v to take care of the black regions after rotation
                u = Reflection(u, imW);
                u = floor(u);
                v = Reflection(v, imH);
                v = floor(v);

                *pCurrentMapping = (int)(v * imW + u);
                pCurrentMapping++;
            }
        }
    }
    return TRUE;
}

BOOL CFlatImgFileRotaionFilterBilinear::Init()
{
    CFlatImgFileFilterBilinear::Init();

    const float s_angles[ROTATION_VARIETY] = {-10.0f, -8.0f, -6.0f, -4.0f, -2.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
    int imH = GetImageHeight();
    int imW = GetImageWidth();
    int cy =  imH / 2;
    int cx =  imW / 2;
    for (int i = 0; i < ROTATION_VARIETY; i++)
    {
        int *pCurrentMapping = _ppOffsetMappingBilinear[i] ;
        int *pCurrentMappingProportion = _ppProportionMappingBilinear[i] ;
        float pi = 3.1415926535f;
        float angle = s_angles[i] / 180.0f * pi;
        float cosAngle = cos(angle);
        float sinAngle = sin(angle);
        
        for (int y = 0; y < imH; y++)
        {
            for (int x = 0; x < imW; x++)
            {
                float u = (cosAngle * (float)(x - cx) + sinAngle * (float)(y - cy)) + cx;
                float v = ((0.0f - sinAngle) * (float)(x - cx) + cosAngle * (float)(y - cy)) + cy;
            
                // Reflecting u and v to take care of the black regions after rotation
                u = Reflection(u, imW);
                int u1 = (int)floor(u);
                v = Reflection(v, imH);
                int v1 = (int)floor(v);

                *pCurrentMapping = v1 * imW + u1; // p11
                *(pCurrentMapping+1) = v1 * imW + min(u1+1,imW-1); // p12
                *(pCurrentMapping+2) = min(v1+1,imH-1) * imW + u1; // p21 
                *(pCurrentMapping+3) = min(v1+1,imH-1) * imW + min(u1+1,imW-1); // p22
                pCurrentMapping +=4;
                *pCurrentMappingProportion = (1 - u + u1)*(1 - v + v1) * 1000; //p11
                *(pCurrentMappingProportion+1) = (u - u1)*(1 - v + v1) * 1000; // p12
                *(pCurrentMappingProportion+2) = (1 - u + u1)*(v - v1) * 1000; // p21
                *(pCurrentMappingProportion+3) = (u - u1)*(v - v1) * 1000; // p22
                pCurrentMappingProportion +=4;
            }
        }
    }
    return TRUE;
}


BOOL CFlatImgFileReflectionFilter::Init()
{
    CFlatImgFileFilterBase::Init();

    int *pCurrentMapping = _ppOffsetMapping[0];
    for (int y = 0; y < GetImageHeight(); y++)
    {
        for (int x = 0; x < GetImageWidth(); x++)
        {
            *pCurrentMapping = y * GetImageWidth() + GetImageWidth() - x - 1;
            pCurrentMapping++;
        }
    }
    return TRUE;
}

BOOL CFlatImgFileZoomingFilter::Init()
{
    CFlatImgFileFilterBase::Init();

    const float s_zoomRatio[ZOOMING_VARIETY] = {1.1f, 1.2f, 0.9f, 0.8f};
    int imH = GetImageHeight();
    int imW = GetImageWidth();
    int cy =  imH / 2;
    int cx =  imW / 2;
    for (int i = 0; i < ZOOMING_VARIETY; i++)
    {
        int *pCurrentMapping = _ppOffsetMapping[i] ;
        float ratioInv = 1.0f/s_zoomRatio[i]; // because 0.8 != 1/1.2, I know does not matter much, but just to be accurate
        for (int y = 0; y < imH; y++)
        {
            for (int x = 0; x < imW; x++)
            {
                float u = (ratioInv * (float)(x - cx)) + cx;
                float v = (ratioInv * (float)(y - cy)) + cy;

                // Reflecting u and v to take care of the black regions after zooming if any
                u = Reflection(u, imW);
                u = floor(u);
                v = Reflection(v, imH);
                v = floor(v);
                *pCurrentMapping = (int)v * imW + (int)u;
                pCurrentMapping++;
            }
        }
    }
    return TRUE;
}

BOOL CFlatImgFileZoomingFilterBilinear::Init()
{
    CFlatImgFileFilterBilinear::Init();

    const float s_zoomRatio[ZOOMING_VARIETY] = {1.1f, 1.2f, 0.9f, 0.8f};
    int imH = GetImageHeight();
    int imW = GetImageWidth();
    int cy =  imH / 2;
    int cx =  imW / 2;
    for (int i = 0; i < ZOOMING_VARIETY; i++)
    {
        int *pCurrentMapping = _ppOffsetMappingBilinear[i] ;
        int *pCurrentMappingProportion = _ppProportionMappingBilinear[i] ;
        float ratioInv = 1.0f/s_zoomRatio[i];
        for (int y = 0; y < imH; y++)
        {
            for (int x = 0; x < imW; x++)
            {
                float u = (ratioInv * (float)(x - cx)) + cx;
                float v = (ratioInv * (float)(y - cy)) + cy;

                // Reflecting u and v to take care of the black regions after zooming if any
                u = Reflection(u, imW);
                int u1 = (int)floor(u);
                v = Reflection(v, imH);
                int v1 = (int)floor(v);

                *pCurrentMapping = v1 * imW + u1; // p11
                *(pCurrentMapping+1) = v1 * imW + min(u1+1,imW-1); // p12
                *(pCurrentMapping+2) = min(v1+1,imH-1) * imW + u1; // p21 
                *(pCurrentMapping+3) = min(v1+1,imH-1) * imW + min(u1+1,imW-1); // p22
                pCurrentMapping +=4;
                *pCurrentMappingProportion = (1 - u + u1)*(1 - v + v1) * 1000; //p11
                *(pCurrentMappingProportion+1) = (u - u1)*(1 - v + v1) * 1000; // p12
                *(pCurrentMappingProportion+2) = (1 - u + u1)*(v - v1) * 1000; // p21
                *(pCurrentMappingProportion+3) = (u - u1)*(v - v1) * 1000; // p22
                pCurrentMappingProportion +=4;
            }
        }
    }
    return TRUE;
}

}
