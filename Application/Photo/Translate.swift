//
//  Translate.swift
//  Photo
//
//  Created by 张力允 on 2020/12/16.
//

import Foundation
import PythonKit

class Translate
{
    private let tokenizer: PythonObject?
    private let model: PythonObject?
    private let sys: PythonObject
    private let translateFile: PythonObject
    private let path = "/Users/zhangliyun/Developer/CXSJ3/opus-mt-en-zh"
    
    init() throws
    {
        sys = Python.import("sys")
        sys.path.append("/Users/zhangliyun/Developer/CXSJ3")
        translateFile = Python.import("Translate")
        
        let os = Python.import("os")
        if let flag = Bool(os.path.exists(path)), flag
        {
            tokenizer = translateFile.load_tokenizer(path)
            model = translateFile.load_model(path)
        }
        else
        {
            throw myErrors.fileNotExists("File Doesn't Exist", "Translation model dosen't exist, please contact the developer. ")
        }
    }
    
    func translate(_ text: String) -> String?
    {
        if let tokenizer = self.tokenizer, let model = self.model
        {
            return Array(translateFile.translate(tokenizer, model, text))?[0]
        }
        return nil
    }
}
