
def genrateSendOTP(email_id):
    import random
    import smtplib

    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login('dikshantmali43@gmail.com','######')
    genrateSendOTP.otp = ''.join([str(random.randint(0,9)) for i in range(4)])
    msg='Hello, Your OTP is '+genrateSendOTP.otp
    print(msg)
    genrateSendOTP.otp = int(genrateSendOTP.otp)
    server.sendmail('dikshantmali43@gmail.com',email_id,msg)
    server.quit()

def checkOTP(myotp):
    myotp = int(myotp)
    if genrateSendOTP.otp == myotp:
        return True
    else:
        return False

