//
//  Recognize.swift
//  Photo
//
//  Created by 张力允 on 2020/12/16.
//

import Foundation
import PythonKit

class Recognize
{
    private let detectionFile: PythonObject
    private let recognizationFile: PythonObject
    private let detection: PythonObject?
    private let recognization: PythonObject?
    private let sys = Python.import("sys")
    private let detectionPath = "/Users/zhangliyun/Developer/CXSJ3/detection/craft.pth"
    private let recognizationPath = "/Users/zhangliyun/Developer/CXSJ3/recognization/crnn.pth"
    
    init() throws
    {
        sys.path.append("/Users/zhangliyun/Developer/CXSJ3")
        detectionFile = Python.import("DetectText")
        recognizationFile = Python.import("RecognizeText")
        let os = Python.import("os")
        
        if let flag = Bool(os.path.exists(detectionPath)), flag
        {
            detection = detectionFile.load_model(detectionPath)
        }
        else
        {
            throw myErrors.fileNotExists("File Doesn't Exist", "Detection model dosen't exist, please contact the developer. ")
        }

        if let flag = Bool(os.path.exists(recognizationPath)), flag
        {
            recognization = recognizationFile.load_model(recognizationPath)
        }
        else
        {
            throw myErrors.fileNotExists("File Doesn't Exist", "Recognization model dosen't exist, please contact the developer. ")
        }
        
    }
    
    func recognize(_ imgSrc: String) -> String?
    {
        if let detection = self.detection, let recognization = self.recognization
        {
            let result = self.detectionFile.detect(detection, recognization, imgSrc)
            return String(result)
        }
        return nil
    }
}
