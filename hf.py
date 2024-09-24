
if False:
    import time
    import pyautogui
    import pyperclip
    ''' 
    print(pyautogui.size())
    pyautogui.moveRel(0, -400, duration = 1)
    pyautogui.click()
    pyautogui.keyDown("ctrlleft")
    pyautogui.keyDown("a")
    pyautogui.keyUp("ctrlleft")
    pyautogui.keyUp("a")''' 
    sec_delay = 0.5
    total_time=0.5 # hrs
    count = 60/sec_delay*60*total_time*100

    #count=10
    i=0
    print(count)
    for j in range(int(count)):
        i=i+1
        '''if i % 40 == 0:
            #write to disk
            pyautogui.keyDown("ctrlleft")
            pyautogui.keyDown("c")
            time.sleep(0.1)
            pyautogui.keyUp("ctrlleft")
            pyautogui.keyUp("c")
            time.sleep(0.1)
            a1=pyperclip.paste()
            f = open("text"+str(i)+".txt", "w", encoding="utf-8")
            f.write(a1)
            f.close()
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.keyDown("ctrlleft")
            pyautogui.keyDown("a")
            time.sleep(0.1)
            pyautogui.keyUp("ctrlleft")
            pyautogui.keyUp("a")'''
        pyautogui.scroll(400)
        #sleep for 0.5 seconds 
        time.sleep(1)
        

    pyautogui.keyDown("ctrlleft")
    pyautogui.keyDown("c")
    pyautogui.keyUp("ctrlleft")
    pyautogui.keyUp("c")
    a1=pyperclip.paste()

    #write to disk
    f = open("text.txt", "w", encoding="utf-8")
    f.write(a1)
    f.close()

    #pyautogui.scroll(10000)
    #pyautogui.typewrite("hello Geeks !")


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)