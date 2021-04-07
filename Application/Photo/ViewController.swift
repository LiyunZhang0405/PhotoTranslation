//
//  ViewController.swift
//  Photo
//
//  Created by 张力允 on 2020/12/15.
//

import Cocoa
import PythonKit

class ViewController: NSViewController {

    
    @IBOutlet weak var ocrResult: NSTextField!
    @IBOutlet weak var imgSrc: NSTextField!
    @IBOutlet weak var translateResult: NSTextField!
    @IBOutlet weak var imageView: NSImageView!
    
    var recognize: Recognize?
    var translate: Translate?
    
    func warning(message: String, information: String)
    {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = information
        alert.alertStyle = .informational
        alert.runModal()
    }
        
    @IBAction func checkImage(_ sender: NSButton)
    {
        let url = URL(fileURLWithPath: imgSrc.stringValue)
        if let data = try? Data(contentsOf: url)
        {
            let image = NSImage(data: data)
            imageView.image = image
        }
        else
        {
            warning(message: "Address Error",
                    information: "No such image exists, please enter the correct address")
        }
    }
    
    @IBAction func recognize(_ sender: NSButton)
    {
        ocrResult.stringValue = "Recognizing..."
        if let result = recognize?.recognize(imgSrc.stringValue)
        {
            self.ocrResult.stringValue = result
        }
        else
        {
            warning(message: "Recognization Error", information: "Cannot Recognize the image, please change another one")
        }
    }
    
    @IBAction func translate(_ sender: NSButton)
    {
        let result = translate?.translate(ocrResult.stringValue)
        if let text = result
        {
            translateResult.stringValue = text
        }
        else
        {
            warning(message: "Translation Error", information: "Cannot Translate the text, please check its content is correct")
        }
    }
    
    override func viewWillAppear()
    {
        super.viewDidLoad()
        ocrResult.placeholderString = "Recognized result will be shown here"
        imgSrc.placeholderString = "Input the address of image here"
        translateResult.placeholderString = "Translated resulte will be shown here"
        imgSrc.becomeFirstResponder()
    }
    
    override func viewDidLoad()
    {
        do
        {
            recognize = try Recognize()
            translate = try Translate()
        }
        catch myErrors.fileNotExists(let message, let information)
        {
            warning(message: message, information: information)
        }
        catch { }
    }

}

